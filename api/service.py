"""service.py — lapisan yang memegang model dan menyusun jawaban.

Pembagian tugas di folder ``api/``:

* ``schemas.py``  — apa yang boleh masuk dan keluar (validasi)
* ``service.py``  — siklus hidup model & isi jawaban  <-- file ini
* ``main.py``     — routing HTTP, middleware, penanganan error

Pemisahan ini bukan formalitas. Logika prediksi yang menempel di fungsi
endpoint hanya bisa diuji lewat HTTP; dipisah begini, ia bisa diuji langsung
dan dipakai ulang oleh Streamlit maupun job batch tanpa menyalakan server.

Yang TIDAK ada di sini: perhitungan preprocessing. Itu semua tinggal di
``src/models/predict.py`` — satu jalur inference, dipakai CLI maupun API.
Menulis ulang jalur itu di sisi API adalah cara paling umum melahirkan
training-serving skew.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from src.models.predict import (
    CONTOH_REQUEST,
    ModelNotFoundError,
    load_bundle,
    predict_one,
)
from src.utils import config, metrics, tracking
from src.utils.logger import get_logger

logger = get_logger("house_price.api")

# z untuk interval 80% dua sisi. 80%, bukan 95%: interval 95% pada model ini
# lebarnya hampir ±40% dan berhenti berguna untuk mengambil keputusan.
Z_80 = 1.2816

# Metrik yang layak dipamerkan di /model-info.
_METRIK = (
    "cv_rmse_log", "cv_rmse_log_std", "cv_mae_usd", "cv_r2",
    "holdout_rmse_log", "holdout_mae_usd", "holdout_r2", "holdout_rows",
    "holdout_cv_gap", "holdout_cv_gap_in_std",
)


# ===========================================================================
# PERGANTIAN MODEL OTOMATIS (automatic model switching)
#
# Bedakan dua hal yang sering tertukar:
#
#   Automatic PROMOTION  — MLflow memindahkan alias @champion ke versi baru,
#                          setelah quality gate & perbandingan champion.
#                          Itu terjadi di src/models/trainer.py.
#   Automatic SWITCHING  — API SADAR alias sudah berpindah, lalu mengganti
#                          objek model yang ada di RAM.  <-- bagian ini.
#
# API tidak pernah memutuskan model mana yang menang. Ia hanya MENGIKUTI apa
# pun yang sudah menjadi @champion. Sumber keputusan tetap quality gate.
# ===========================================================================


class KontrakTidakCocok(Exception):
    """Model kandidat meminta field yang berbeda dari kontrak API yang berlaku."""


class ModelKandidatTidakValid(Exception):
    """Model kandidat gagal uji coba prediksi — tidak layak menggantikan yang aktif."""


# --- Keadaan bersama, dijaga dua kunci dengan tugas berbeda ---------------
#
# _KUNCI_STATE dipegang hanya sepersekian milidetik (baca/tulis dua variabel),
# jadi /predict nyaris tidak pernah menunggu.
# _KUNCI_MUAT dipegang lama (unduh + muat + validasi), tapi TIDAK dipegang oleh
# jalur prediksi — gunanya cuma mencegah dua reload berjalan bersamaan.
#
# Kalau keduanya digabung jadi satu kunci, tiap request prediksi akan antre di
# belakang proses unduh model. Itu persis race condition yang tidak terlihat
# saat diuji sendirian dan baru muncul saat trafik ramai.
_KUNCI_STATE = threading.RLock()
_KUNCI_MUAT = threading.Lock()

_AKTIF: dict[str, Any] | None = None      # bundle yang sedang melayani
_IDENTITAS: str | None = None             # penanda versi bundle yang sedang melayani

_BERHENTI = threading.Event()
_PEMANTAU: threading.Thread | None = None

# Hasil resolusi mode "auto": setelah sekali memuat, kita TAHU sumber mana yang
# sebenarnya bisa dipakai. Ini penting dan halus.
#
# Di mode auto, membaca nomor versi dari registry bisa BERHASIL sementara
# mengunduh artifact-nya GAGAL (persis kasus Docker: nomor versi ada di
# database, tapi path artifact-nya milik mesin lain). Tanpa mengingat hasil
# resolusi ini, pemeriksaan berkala akan terus melihat "champion = registry:v1"
# sementara yang benar-benar dimuat adalah berkas — beda selamanya, sehingga
# API memuat ulang model setiap kali polling. Reload tanpa henti yang tidak
# memperbaiki apa pun.
_SUMBER_EFEKTIF: str | None = None


def _sumber() -> str:
    return config.get_settings().model_source.strip().lower()


def _bundle_dari_file() -> dict[str, Any]:
    bundle = load_bundle()
    bundle.setdefault("sumber_model", {"sumber": "file",
                                       "path": str(config.MODEL_ARTIFACT)})
    return bundle


def _identitas_file() -> str:
    """Penanda versi untuk artifact berkas: waktu ubah + ukuran.

    Berkas tidak punya nomor versi seperti registry, jadi yang dipakai adalah
    sidik jari murah yang berubah begitu trainer menulis ulang artifact-nya.
    Ini yang membuat pergantian otomatis TETAP BEKERJA di dalam container,
    tempat registry MLflow tidak terjangkau (lihat catatan Docker di README).
    """
    st = config.MODEL_ARTIFACT.stat()
    return f"file:{st.st_mtime_ns}:{st.st_size}"


def _identitas_registry() -> str:
    return f"registry:v{tracking.versi_champion()}"


def _identitas_dari_bundle(bundle: dict[str, Any]) -> str:
    """Penanda versi yang menggambarkan bundle yang BENAR-BENAR dimuat.

    Identitas selalu diturunkan dari bundle hasil pemuatan, bukan dari
    perkiraan sebelum memuat — supaya "apa yang aktif" dan "apa yang dicek"
    selalu diukur dengan penggaris yang sama.
    """
    sumber_model = bundle.get("sumber_model") or {}
    if sumber_model.get("sumber") == "registry":
        return f"registry:v{sumber_model.get('versi')}"
    return _identitas_file()


def identitas_champion() -> str:
    """Penanda versi model yang SEHARUSNYA aktif, menurut sumber kebenaran.

    Sengaja murah: mode registry cuma satu query ke database alias, tanpa
    mengunduh apa pun. Yang mahal baru dijalankan setelah terbukti berbeda.
    """
    sumber = _sumber()
    if sumber == "file":
        return _identitas_file()
    if sumber == "registry":
        return _identitas_registry()

    # Mode auto: pakai hasil resolusi kalau sudah pernah memuat sekali.
    if _SUMBER_EFEKTIF == "file":
        return _identitas_file()
    try:
        return _identitas_registry()
    except Exception:  # noqa: BLE001 - mode auto memang boleh jatuh ke berkas
        return _identitas_file()


def _muat_bundle_baru() -> dict[str, Any]:
    """Muat bundle sesuai MODEL_SOURCE. Boleh mahal — dipanggil saat perlu saja."""
    sumber = _sumber()
    if sumber == "file":
        return _bundle_dari_file()
    if sumber == "registry":
        # Sengaja TIDAK jatuh ke berkas: kalau operator menyatakan registry,
        # diam-diam menyajikan model lain adalah kebohongan tanpa error.
        return tracking.muat_bundle_champion()
    try:
        return tracking.muat_bundle_champion()
    except Exception as err:  # noqa: BLE001
        logger.warning("registry_tidak_terjangkau_pakai_file",
                       extra={"alasan": str(err)[:200]})
        return _bundle_dari_file()


def validasi_bundle(bundle: dict[str, Any]) -> None:
    """Dua pemeriksaan sebelum model kandidat boleh menggantikan yang aktif.

    1. **Kontrak** — field yang diminta model harus sama persis dengan kontrak
       API yang sedang berlaku. Model dengan kontrak berbeda menjawab soal yang
       berbeda; menyajikannya lewat endpoint yang sama berarti user mengirim 20
       field dan model diam-diam membacanya sebagai sesuatu yang lain.
    2. **Uji coba prediksi** — model benar-benar bisa menghasilkan angka wajar.
       Ini yang menangkap artifact rusak atau setengah terunduh, kegagalan yang
       tidak terlihat sampai request pertama masuk.
    """
    kontrak_model = set(bundle.get("api_contract_fields") or [])
    kontrak_api = set(config.API_CONTRACT)
    if kontrak_model != kontrak_api:
        raise KontrakTidakCocok(
            f"kontrak model tidak cocok — kurang={sorted(kontrak_api - kontrak_model)}, "
            f"berlebih={sorted(kontrak_model - kontrak_api)}"
        )

    try:
        hasil = predict_one(CONTOH_REQUEST, bundle, request_id="validasi-kandidat",
                            klien="validasi")
    except Exception as err:  # noqa: BLE001
        raise ModelKandidatTidakValid(f"uji coba prediksi gagal: {err}") from err

    harga = float(hasil["predicted_price"])
    if not (0 < harga < 100_000_000):
        raise ModelKandidatTidakValid(f"prediksi uji coba di luar nalar: {harga}")


def _catat_metrik_versi(identitas: str) -> None:
    """Terjemahkan penanda versi jadi angka untuk Prometheus (0 kalau dari berkas)."""
    nomor = 0.0
    if identitas.startswith("registry:v"):
        try:
            nomor = float(identitas.split("v", 1)[1])
        except ValueError:
            nomor = 0.0
    metrics.model_versi_aktif.set(nomor)


def periksa_dan_ganti(paksa: bool = False) -> dict[str, Any]:
    """Inti pergantian otomatis: cek versi, muat, validasi, lalu ganti — dalam urutan itu.

    Urutannya yang menentukan keselamatan. Model lama TIDAK PERNAH dibuang
    lebih dulu: kandidat dimuat dan diuji sepenuhnya di variabel terpisah, dan
    penggantian baru terjadi setelah semuanya lolos. Kalau ada satu langkah
    gagal, ``_AKTIF`` tidak tersentuh dan API terus melayani dengan model lama.

    Melempar exception kalau gagal — supaya /admin/reload bisa melaporkannya.
    Pemantau latar memakai ``periksa_dan_ganti_aman()`` yang menelannya.
    """
    global _AKTIF, _IDENTITAS, _SUMBER_EFEKTIF

    with _KUNCI_MUAT:
        identitas_baru = identitas_champion()

        with _KUNCI_STATE:
            identitas_lama, ada_model = _IDENTITAS, _AKTIF is not None

        if ada_model and not paksa and identitas_baru == identitas_lama:
            metrics.model_reload_total.labels(hasil="tidak_perlu").inc()
            return {"berganti": False, "alasan": "champion tidak berubah",
                    "identitas": identitas_lama}

        if ada_model and identitas_baru != identitas_lama:
            logger.info("champion_berubah", extra={"dari": identitas_lama,
                                                   "ke": identitas_baru})

        try:
            logger.info("model_reload_mulai", extra={"identitas": identitas_baru,
                                                     "sumber": _sumber()})
            kandidat = _muat_bundle_baru()
            validasi_bundle(kandidat)
        except Exception as err:  # noqa: BLE001 - apa pun sebabnya, model lama dipertahankan
            metrics.model_reload_total.labels(hasil="gagal").inc()
            logger.error("model_reload_gagal",
                         extra={"identitas_kandidat": identitas_baru,
                                "tipe": type(err).__name__, "alasan": str(err)[:300],
                                "model_aktif_tetap": identitas_lama})
            raise

        # Identitas yang DISIMPAN diturunkan dari bundle hasil pemuatan, bukan
        # dari tebakan sebelum memuat — di mode auto keduanya bisa berbeda.
        identitas_baru = _identitas_dari_bundle(kandidat)
        _SUMBER_EFEKTIF = (kandidat.get("sumber_model") or {}).get("sumber", "file")

        with _KUNCI_STATE:
            _AKTIF, _IDENTITAS = kandidat, identitas_baru

        metrics.model_reload_total.labels(hasil="sukses").inc()
        metrics.model_switch_terakhir.set(time.time())
        _catat_metrik_versi(identitas_baru)
        logger.info("model_reload_sukses",
                    extra={"dari": identitas_lama, "ke": identitas_baru,
                           "model": kandidat.get("model_name"),
                           "sumber_model": kandidat.get("sumber_model")})
        return {"berganti": True, "dari": identitas_lama, "ke": identitas_baru,
                "model_name": kandidat.get("model_name"),
                "sumber_model": kandidat.get("sumber_model")}


def periksa_dan_ganti_aman() -> dict[str, Any]:
    """Versi yang tidak pernah melempar — dipakai pemantau latar.

    Pergantian model yang gagal tidak boleh menjatuhkan API. Registry mati,
    jaringan putus, artifact rusak: semuanya berakhir sama — dicatat, dihitung
    di metrik, lalu API terus melayani dengan model lama.
    """
    try:
        return periksa_dan_ganti()
    except Exception as err:  # noqa: BLE001
        return {"berganti": False, "gagal": True, "tipe": type(err).__name__,
                "alasan": str(err)[:300]}


def muat_model() -> dict[str, Any]:
    """Bundle yang sedang melayani. Dipanggil tiap request, jadi harus murah.

    Yang dilakukan cuma membaca satu variabel di balik kunci — tanpa menyentuh
    MLflow, tanpa membaca berkas. Pemeriksaan versi adalah tugas pemantau
    berkala, bukan tugas jalur prediksi: kalau tiap request ikut mengecek
    registry, MLflow jadi bagian dari jalur kritis dan latency-nya ikut kita.
    """
    with _KUNCI_STATE:
        if _AKTIF is not None:
            return _AKTIF

    # Belum pernah dimuat (startup, atau semua percobaan sebelumnya gagal).
    periksa_dan_ganti(paksa=True)
    with _KUNCI_STATE:
        if _AKTIF is None:  # pragma: no cover - hanya kalau ada bug di atas
            raise ModelNotFoundError("model tidak berhasil dimuat")
        return _AKTIF


def siap() -> bool:
    """True kalau model benar-benar sudah bisa melayani request."""
    try:
        muat_model()
        return True
    except Exception:  # noqa: BLE001 - sengaja seluas mungkin
        return False


def muat_ulang() -> dict[str, Any]:
    """Paksa muat ulang, walau versinya tidak berubah — dipakai /admin/reload."""
    load_bundle.cache_clear()
    periksa_dan_ganti(paksa=True)
    return muat_model()


def lupakan_model() -> None:
    """Kosongkan seluruh keadaan model — termasuk hasil resolusi mode auto."""
    global _AKTIF, _IDENTITAS, _SUMBER_EFEKTIF
    load_bundle.cache_clear()
    with _KUNCI_STATE:
        _AKTIF, _IDENTITAS, _SUMBER_EFEKTIF = None, None, None


def versi_singkat() -> str | None:
    """Versi model aktif dalam bentuk pendek untuk /health: "2" atau "file"."""
    with _KUNCI_STATE:
        aktif = _AKTIF
    if aktif is None:
        return None
    sumber_model = aktif.get("sumber_model") or {}
    if sumber_model.get("sumber") == "registry":
        return str(sumber_model.get("versi"))
    return "file"


def status_model() -> dict[str, Any]:
    """Ringkasan aman untuk dilihat manusia: apa yang aktif, apa yang seharusnya.

    Read-only dan tanpa parameter — sengaja. Endpoint status yang menerima
    "muat model dari path X" adalah lubang keamanan, bukan alat bantu.
    """
    with _KUNCI_STATE:
        aktif, identitas = _AKTIF, _IDENTITAS

    try:
        seharusnya = identitas_champion()
        catatan = None
    except Exception as err:  # noqa: BLE001
        seharusnya, catatan = None, f"sumber kebenaran tidak terjangkau: {str(err)[:200]}"

    return {
        "sumber": _sumber(),
        "model_dimuat": aktif is not None,
        "model_name": None if aktif is None else aktif.get("model_name"),
        "identitas_aktif": identitas,
        "identitas_champion": seharusnya,
        "perlu_ganti": bool(seharusnya and identitas and seharusnya != identitas),
        "sumber_model": None if aktif is None else aktif.get("sumber_model"),
        "interval_pemeriksaan_detik": config.get_settings().model_check_interval_seconds,
        "pemantau_aktif": bool(_PEMANTAU and _PEMANTAU.is_alive()),
        "catatan": catatan,
    }


# --- Pemantau berkala -----------------------------------------------------
def _loop_pemantau(interval: int) -> None:
    """Loop latar: tidur, cek, ulangi. Sengaja sesederhana mungkin.

    Memakai Event.wait() dan bukan time.sleep() supaya thread ini bisa
    dihentikan seketika saat aplikasi mati — tidak menahan proses shutdown
    sampai tidurnya habis.
    """
    logger.info("pemantau_champion_mulai", extra={"interval_detik": interval})
    while not _BERHENTI.wait(interval):
        hasil = periksa_dan_ganti_aman()
        if hasil.get("berganti"):
            logger.info("model_berganti_otomatis", extra=hasil)
    logger.info("pemantau_champion_berhenti")


def mulai_pemantau() -> bool:
    """Nyalakan pemantau kalau MODEL_CHECK_INTERVAL_SECONDS > 0."""
    global _PEMANTAU
    interval = int(config.get_settings().model_check_interval_seconds)
    if interval <= 0:
        logger.info("pemantau_champion_nonaktif", extra={"interval_detik": interval})
        return False
    if _PEMANTAU is not None and _PEMANTAU.is_alive():
        return True

    _BERHENTI.clear()
    _PEMANTAU = threading.Thread(target=_loop_pemantau, args=(interval,),
                                 name="pemantau-champion", daemon=True)
    _PEMANTAU.start()
    return True


def hentikan_pemantau() -> None:
    """Hentikan pemantau dengan rapi saat aplikasi mati."""
    global _PEMANTAU
    _BERHENTI.set()
    if _PEMANTAU is not None:
        _PEMANTAU.join(timeout=5)
        _PEMANTAU = None


def _interval(harga: float, rmse_log: float | None) -> dict[str, float]:
    """Interval 80% dari sebaran error di skala log.

    Cara bacanya jujur: ini perkiraan, bukan jaminan. Asumsinya error di skala
    log tersebar merata untuk semua rentang harga — asumsi yang cukup baik di
    sini karena ``log1p`` memang dipilih untuk meratakannya, tapi tetap asumsi.
    Rumah yang sangat mahal cenderung punya error lebih besar dari yang
    tergambar di sini.
    """
    if not rmse_log:
        return {"bawah": harga, "atas": harga, "tingkat_keyakinan": 0.0}
    log_harga = float(np.log1p(harga))
    return {
        "bawah": round(float(np.expm1(log_harga - Z_80 * rmse_log)), 2),
        "atas": round(float(np.expm1(log_harga + Z_80 * rmse_log)), 2),
        "tingkat_keyakinan": 0.8,
    }


def prediksi(payload: dict[str, Any], request_id: str,
             klien: str = "produksi") -> dict[str, Any]:
    """Satu rumah -> satu jawaban lengkap dengan interval & identitas model."""
    bundle = muat_model()
    try:
        hasil = predict_one(payload, bundle, request_id=request_id, klien=klien)
    except Exception:
        metrics.prediksi_total.labels(hasil="gagal").inc()
        raise

    # Metrik dicatat DI SINI, bukan di endpoint, supaya pemakai lain jalur ini
    # (job batch, Streamlit) ikut terhitung — bukan cuma yang lewat HTTP.
    metrics.prediksi_total.labels(hasil="sukses").inc()
    metrics.prediksi_durasi.observe(hasil["latency_ms"] / 1000.0)
    metrics.prediksi_harga_usd.observe(hasil["predicted_price"])

    return {
        "request_id": request_id,
        "predicted_price_usd": hasil["predicted_price"],
        "interval": _interval(hasil["predicted_price"], bundle.get("holdout_rmse_log")),
        "currency": "USD",
        "model_name": hasil["model_name"],
        "model_type": hasil["model_type"],
        "model_trained_on_rows": int(bundle.get("training_rows", 0)),
        "latency_ms": hasil["latency_ms"],
    }


def prediksi_batch(rows: list[dict[str, Any]], request_id: str,
                   klien: str = "produksi") -> dict[str, Any]:
    """Banyak rumah sekaligus. Tiap baris tetap punya request_id turunan sendiri.

    Kenapa tidak satu id untuk seluruh batch? Karena saat menelusuri log nanti,
    yang dicari adalah satu prediksi yang mencurigakan — bukan seluruh batch.
    """
    mulai = time.perf_counter()
    hasil = [prediksi(baris, f"{request_id}-{i}", klien)
             for i, baris in enumerate(rows, start=1)]
    return {
        "request_id": request_id,
        "count": len(hasil),
        "predictions": hasil,
        "latency_ms": round((time.perf_counter() - mulai) * 1000, 2),
    }


def info_model() -> dict[str, Any]:
    """Isi /model-info: siapa modelnya, dilatih pada apa, seberapa bagus."""
    bundle = muat_model()
    return {
        "model_name": bundle.get("model_name", "tidak-diketahui"),
        "model_type": bundle.get("model_type", "tidak-diketahui"),
        "target": bundle.get("target", config.TARGET),
        "target_transformation": bundle.get("target_transformation", "-"),
        "training_rows": int(bundle.get("training_rows", 0)),
        "created_at": str(bundle.get("created_at", "-")),
        "api_contract_fields": list(bundle.get("api_contract_fields", [])),
        "engineered_features": list(bundle.get("engineered_features", [])),
        "leakage_features_removed": list(bundle.get("leakage_features_removed", [])),
        "sumber_model": bundle.get("sumber_model"),
        "metrics": {k: float(bundle[k]) for k in _METRIK if k in bundle},
        "metrics_note": str(bundle.get("metrics_note", "")),
    }
