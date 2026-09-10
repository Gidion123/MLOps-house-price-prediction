"""Uji pemilihan sumber model — MODEL_SOURCE=file | registry | auto.

Ini setelan yang menentukan model mana yang benar-benar melayani request.
Sebelum ini ia cuma dideklarasikan di config dan tidak pernah dibaca kode mana
pun — setelan mati yang terlihat seperti fitur. Test di sini yang menjaga ia
tetap hidup.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.service as service
from api.main import app
from src.models.predict import load_bundle
from src.utils import config, tracking


def bundle_registry_palsu() -> dict:
    """Bundle yang BERASAL dari artifact sungguhan, cuma ditandai datang dari registry.

    Dulu tiruan di sini berupa dict kosong tanpa preprocessor/model. Itu lolos
    karena waktu itu belum ada validasi kandidat. Begitu ``validasi_bundle``
    menjalankan uji coba prediksi sungguhan, tiruan kosong langsung ditolak —
    dan itu memang yang seharusnya terjadi: model yang tidak bisa memprediksi
    tidak boleh menggantikan model produksi.
    """
    bundle = dict(load_bundle())
    bundle["model_name"] = "model-dari-registry"
    bundle["sumber_model"] = {"sumber": "registry", "nama_model": "house-price-regressor",
                              "versi": 7, "alias": "champion"}
    return bundle


@pytest.fixture(autouse=True)
def bersihkan_cache():
    """Keadaan model wajib dikosongkan sebelum & sesudah tiap uji sumber."""
    service.lupakan_model()
    yield
    service.lupakan_model()


def _atur_sumber(monkeypatch, nilai: str) -> None:
    monkeypatch.setattr(config.get_settings(), "model_source", nilai)


def test_sumber_file_membaca_artifact_joblib(monkeypatch):
    _atur_sumber(monkeypatch, "file")
    bundle = service.muat_model()
    assert bundle["sumber_model"]["sumber"] == "file"
    assert bundle["training_rows"] == 1460


def test_sumber_registry_memakai_alias_champion(monkeypatch):
    _atur_sumber(monkeypatch, "registry")
    monkeypatch.setattr(tracking, "muat_bundle_champion", lambda *a, **k: bundle_registry_palsu())
    monkeypatch.setattr(tracking, "versi_champion", lambda *a, **k: "7")
    bundle = service.muat_model()
    assert bundle["sumber_model"]["sumber"] == "registry"
    assert bundle["sumber_model"]["versi"] == 7
    assert bundle["model_name"] == "model-dari-registry"


def test_sumber_registry_TIDAK_diam_diam_jatuh_ke_file(monkeypatch):
    """Kalau operator menyatakan registry, menyajikan model lain adalah kebohongan.

    Lebih baik gagal keras dan terlihat di /ready daripada melayani artifact
    berbeda tanpa ada yang tahu.
    """
    _atur_sumber(monkeypatch, "registry")

    def meledak(*a, **k):
        raise RuntimeError("registry mati")

    monkeypatch.setattr(tracking, "muat_bundle_champion", meledak)
    with pytest.raises(RuntimeError):
        service.muat_model()


def test_sumber_auto_jatuh_ke_file_kalau_registry_mati(monkeypatch):
    _atur_sumber(monkeypatch, "auto")

    def meledak(*a, **k):
        raise RuntimeError("registry mati")

    monkeypatch.setattr(tracking, "muat_bundle_champion", meledak)
    bundle = service.muat_model()
    assert bundle["sumber_model"]["sumber"] == "file"


def test_sumber_auto_memilih_registry_kalau_hidup(monkeypatch):
    _atur_sumber(monkeypatch, "auto")
    monkeypatch.setattr(tracking, "muat_bundle_champion", lambda *a, **k: bundle_registry_palsu())
    monkeypatch.setattr(tracking, "versi_champion", lambda *a, **k: "7")
    assert service.muat_model()["sumber_model"]["sumber"] == "registry"


def test_model_info_menyebut_sumbernya(monkeypatch):
    """Dua container dengan tag image sama bisa menyajikan versi berbeda —
    jadi 'prediksi ini dari model mana?' wajib punya jawaban."""
    _atur_sumber(monkeypatch, "file")
    with TestClient(app) as c:
        c.headers["X-Traffic-Source"] = "test"
        body = c.get("/model-info").json()
    assert body["sumber_model"]["sumber"] == "file"


def test_muat_ulang_mengambil_bundle_terbaru(monkeypatch):
    _atur_sumber(monkeypatch, "file")
    service.muat_model()
    monkeypatch.setattr(tracking, "muat_bundle_champion", lambda *a, **k: bundle_registry_palsu())
    monkeypatch.setattr(tracking, "versi_champion", lambda *a, **k: "7")
    _atur_sumber(monkeypatch, "registry")
    # tanpa muat ulang, model yang sudah aktif tetap dipakai
    assert service.muat_model()["sumber_model"]["sumber"] == "file"
    assert service.muat_ulang()["sumber_model"]["sumber"] == "registry"


def test_endpoint_rollback_mati_tanpa_token():
    with TestClient(app) as c:
        c.headers["X-Traffic-Source"] = "test"
        assert c.post("/admin/rollback").status_code == 503
