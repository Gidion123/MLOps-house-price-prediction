"""Uji PERGANTIAN MODEL OTOMATIS — dan terutama: apa yang terjadi saat gagal.

Bedanya dua istilah yang sering tertukar:

* **Automatic promotion** — MLflow memindahkan alias ``@champion`` ke versi
  baru setelah quality gate. Diuji di ``test_tracking.py``.
* **Automatic switching** — API menyadari alias sudah berpindah lalu mengganti
  objek model di RAM. Itu yang diuji di sini.

Yang paling penting bukan jalan bahagianya. Fitur ini menyentuh satu-satunya
benda yang membuat API berguna — model yang sedang melayani — jadi pertanyaan
sesungguhnya adalah: **kalau model baru bermasalah, apakah model lama selamat?**
Lima dari delapan uji di bawah menjawab pertanyaan itu.
"""

from __future__ import annotations

import threading
import time

import pytest

import api.service as service
from src.models.predict import CONTOH_REQUEST, load_bundle
from src.utils import config, tracking


@pytest.fixture(scope="module")
def bundle_asli() -> dict:
    """Bundle sungguhan — dipakai sebagai bahan dasar semua model palsu.

    Sengaja model asli, bukan tiruan: validasi kandidat menjalankan uji coba
    prediksi sungguhan, dan tiruan akan melewatinya tanpa membuktikan apa pun.
    """
    return dict(load_bundle())


def bundle_versi(asli: dict, versi: int, kontrak: list[str] | None = None) -> dict:
    b = dict(asli)
    b["sumber_model"] = {"sumber": "registry", "nama_model": "house-price-regressor",
                         "versi": versi, "alias": "champion"}
    if kontrak is not None:
        b["api_contract_fields"] = kontrak
    return b


@pytest.fixture
def registry(monkeypatch, bundle_asli):
    """Registry tiruan yang bisa digeser versinya sesuka uji.

    ``registry.versi`` = versi yang dipegang @champion sekarang.
    ``registry.gagal_muat`` = paksa pemuatan artifact gagal.
    ``registry.mati`` = paksa MLflow tidak terjangkau sama sekali.
    """
    class Palsu:
        versi = 1
        kontrak = None
        gagal_muat = False
        mati = False

    palsu = Palsu()
    monkeypatch.setattr(config.get_settings(), "model_source", "registry")

    def versi_champion(*a, **k):
        if palsu.mati:
            raise RuntimeError("MLflow tidak terjangkau")
        return str(palsu.versi)

    def muat_bundle(*a, **k):
        if palsu.mati:
            raise RuntimeError("MLflow tidak terjangkau")
        if palsu.gagal_muat:
            raise RuntimeError("artifact gagal diunduh")
        return bundle_versi(bundle_asli, palsu.versi, palsu.kontrak)

    monkeypatch.setattr(tracking, "versi_champion", versi_champion)
    monkeypatch.setattr(tracking, "muat_bundle_champion", muat_bundle)

    service.lupakan_model()
    yield palsu
    service.hentikan_pemantau()
    service.lupakan_model()


def versi_aktif() -> int | None:
    sumber = service.status_model()["sumber_model"]
    return None if sumber is None else sumber.get("versi")


# ---------------------------------------------------------------------------
# TEST 1 — champion tidak berubah -> tidak ada reload
# ---------------------------------------------------------------------------
def test_champion_sama_tidak_memicu_reload(registry):
    service.muat_model()
    assert versi_aktif() == 1

    hasil = service.periksa_dan_ganti()
    assert hasil["berganti"] is False
    assert hasil["alasan"] == "champion tidak berubah"
    assert versi_aktif() == 1


# ---------------------------------------------------------------------------
# TEST 2 & 3 — champion berpindah -> reload, dan v2 jadi model aktif
# ---------------------------------------------------------------------------
def test_champion_berubah_memicu_reload_dan_v2_jadi_aktif(registry):
    service.muat_model()
    assert versi_aktif() == 1

    registry.versi = 2
    hasil = service.periksa_dan_ganti()

    assert hasil["berganti"] is True
    assert hasil["dari"] == "registry:v1" and hasil["ke"] == "registry:v2"
    assert versi_aktif() == 2
    assert service.status_model()["perlu_ganti"] is False


# ---------------------------------------------------------------------------
# TEST 4 — kontrak berbeda -> DITOLAK, v1 tetap aktif
# ---------------------------------------------------------------------------
def test_kontrak_berbeda_ditolak_dan_model_lama_bertahan(registry):
    service.muat_model()
    assert versi_aktif() == 1

    registry.versi = 2
    registry.kontrak = ["GrLivArea", "OverallQual"]      # kontrak sempit, tidak cocok

    with pytest.raises(service.KontrakTidakCocok):
        service.periksa_dan_ganti()

    assert versi_aktif() == 1, "model lama harus tetap melayani"
    assert service.muat_model()["api_contract_fields"] == list(config.API_CONTRACT)


# ---------------------------------------------------------------------------
# TEST 5 — model kandidat gagal dimuat -> v1 tetap aktif
# ---------------------------------------------------------------------------
def test_kandidat_gagal_dimuat_model_lama_bertahan(registry):
    service.muat_model()
    registry.versi = 2
    registry.gagal_muat = True

    with pytest.raises(RuntimeError):
        service.periksa_dan_ganti()

    assert versi_aktif() == 1
    # dan API tetap bisa memprediksi dengan model lama
    assert service.prediksi(CONTOH_REQUEST, "uji", klien="test")["predicted_price_usd"] > 0


# ---------------------------------------------------------------------------
# TEST 6 — MLflow mati total -> v1 tetap aktif, API tidak ikut mati
# ---------------------------------------------------------------------------
def test_mlflow_mati_tidak_menjatuhkan_api(registry):
    service.muat_model()
    registry.mati = True

    hasil = service.periksa_dan_ganti_aman()      # versi yang dipakai pemantau latar
    assert hasil["berganti"] is False and hasil["gagal"] is True

    assert versi_aktif() == 1
    assert service.prediksi(CONTOH_REQUEST, "uji", klien="test")["predicted_price_usd"] > 0
    assert service.siap() is True


def test_kandidat_rusak_ditolak_uji_coba_prediksi(registry, bundle_asli):
    """Artifact yang termuat tapi tidak bisa memprediksi juga harus ditolak."""
    rusak = bundle_versi(bundle_asli, 2)
    rusak["model"] = object()                      # bukan model sungguhan

    service.muat_model()
    registry.versi = 2
    import src.utils.tracking as t
    t.muat_bundle_champion = lambda *a, **k: rusak

    with pytest.raises(service.ModelKandidatTidakValid):
        service.periksa_dan_ganti()
    assert versi_aktif() == 1


# ---------------------------------------------------------------------------
# TEST 7 — prediksi bersamaan dengan reload -> tidak ada race condition
# ---------------------------------------------------------------------------
def test_prediksi_dan_reload_bersamaan_aman(registry):
    """Puluhan prediksi berjalan sementara model diganti berkali-kali.

    Yang dicari bukan kecepatan, tapi apakah ada request yang melihat keadaan
    setengah-jadi: model sudah diganti tapi identitasnya belum, atau sebaliknya.
    Itulah yang dijaga _KUNCI_STATE.
    """
    service.muat_model()
    kesalahan: list[Exception] = []
    harga: list[float] = []
    berhenti = threading.Event()

    def prediksi_terus():
        try:
            while not berhenti.is_set():
                harga.append(service.prediksi(CONTOH_REQUEST, "uji-paralel", klien="test")["predicted_price_usd"])
        except Exception as err:  # noqa: BLE001
            kesalahan.append(err)

    def ganti_terus():
        try:
            for versi in range(2, 8):
                registry.versi = versi
                service.periksa_dan_ganti()
                time.sleep(0.01)
        except Exception as err:  # noqa: BLE001
            kesalahan.append(err)
        finally:
            berhenti.set()

    pembaca = [threading.Thread(target=prediksi_terus) for _ in range(4)]
    penulis = threading.Thread(target=ganti_terus)
    for t in pembaca:
        t.start()
    penulis.start()
    penulis.join(timeout=30)
    berhenti.set()
    for t in pembaca:
        t.join(timeout=10)

    assert not kesalahan, f"ada kesalahan saat prediksi & reload bersamaan: {kesalahan}"
    assert len(harga) > 20, "prediksi tidak berjalan selama reload"
    assert all(h > 0 for h in harga)
    assert versi_aktif() == 7


# ---------------------------------------------------------------------------
# Pemantau latar — bukti pergantian benar-benar OTOMATIS, bukan cuma manual
# ---------------------------------------------------------------------------
def test_pemantau_latar_mengganti_model_sendiri(registry, monkeypatch):
    monkeypatch.setattr(config.get_settings(), "model_check_interval_seconds", 1)
    service.muat_model()
    assert versi_aktif() == 1

    assert service.mulai_pemantau() is True
    assert service.status_model()["pemantau_aktif"] is True

    registry.versi = 2                       # simulasi promosi di MLflow
    batas = time.time() + 15
    while time.time() < batas and versi_aktif() != 2:
        time.sleep(0.2)

    assert versi_aktif() == 2, "pemantau latar tidak mengganti model dalam 15 detik"
    service.hentikan_pemantau()
    assert service.status_model()["pemantau_aktif"] is False


def test_pemantau_mati_kalau_interval_nol(registry, monkeypatch):
    monkeypatch.setattr(config.get_settings(), "model_check_interval_seconds", 0)
    assert service.mulai_pemantau() is False
    assert service.status_model()["pemantau_aktif"] is False
