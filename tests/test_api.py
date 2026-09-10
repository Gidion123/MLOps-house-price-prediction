"""Uji API — yang diperiksa di sini adalah PENOLAKAN, bukan cuma jalan bahagia.

Endpoint yang benar mudah dibuat. Yang sulit — dan yang menentukan apakah ini
layak disebut production — adalah apa yang terjadi saat request-nya salah:
field kurang, field asing, angka di luar nalar, kategori yang tidak ada.
Semua itu harus ditolak 422 dengan pesan yang menyebut field-nya, bukan
diterima lalu dijawab HTTP 200 dengan angka ngawur.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import BATAS_BATCH, HousePriceRequest
from src.models.predict import CONTOH_REQUEST
from src.utils import config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # Menandai seluruh trafik test sebagai sintetis. Tanpa ini, tiap
        # `pytest` menambahkan ratusan request berisi rumah yang sama persis
        # ke jendela analisis drift — dan laporan drift berikutnya berbunyi
        # karena kita sendiri, bukan karena user.
        c.headers["X-Traffic-Source"] = "test"
        yield c


# ---------------------------------------------------------------------------
# Meta & kesiapan
# ---------------------------------------------------------------------------
def test_health_selalu_menjawab(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_ready_ok_saat_model_ada(client):
    assert client.get("/ready").status_code == 200


def test_model_info_menyebut_identitas_dan_metrik(client):
    body = client.get("/model-info").json()
    assert body["model_type"] == "XGBRegressor"
    assert body["training_rows"] == 1460
    assert len(body["api_contract_fields"]) == len(config.API_CONTRACT)
    assert body["metrics"]["holdout_r2"] > 0.85
    # Catatan kejujuran wajib ikut: artifact bukan model yang menghasilkan
    # angka holdout, dan itu harus terbaca siapa pun yang mengaudit.
    assert "dilatih ulang" in body["metrics_note"]


def test_request_id_dikembalikan_di_header(client):
    r = client.post("/predict", json=CONTOH_REQUEST, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["request_id"] == "abc123"


# ---------------------------------------------------------------------------
# Jalan bahagia
# ---------------------------------------------------------------------------
def test_predict_menghasilkan_harga_masuk_akal(client):
    body = client.post("/predict", json=CONTOH_REQUEST).json()
    assert 50_000 < body["predicted_price_usd"] < 800_000
    assert body["currency"] == "USD"
    assert body["model_trained_on_rows"] == 1460
    assert body["latency_ms"] > 0


def test_interval_mengapit_prediksi(client):
    body = client.post("/predict", json=CONTOH_REQUEST).json()
    harga, iv = body["predicted_price_usd"], body["interval"]
    assert iv["bawah"] < harga < iv["atas"]
    assert iv["tingkat_keyakinan"] == 0.8


def test_batch_menjawab_sejumlah_baris(client):
    r = client.post("/predict/batch", json={"houses": [CONTOH_REQUEST] * 3})
    body = r.json()
    assert r.status_code == 200
    assert body["count"] == 3 and len(body["predictions"]) == 3
    # id tiap baris harus unik supaya bisa ditelusuri satu per satu
    assert len({p["request_id"] for p in body["predictions"]}) == 3


# ---------------------------------------------------------------------------
# Penolakan — bagian yang paling penting
# ---------------------------------------------------------------------------
def test_field_kurang_ditolak(client):
    kurang = {k: v for k, v in CONTOH_REQUEST.items() if k != "GrLivArea"}
    r = client.post("/predict", json=kurang)
    assert r.status_code == 422
    assert "GrLivArea" in str(r.json()["detail"])


def test_field_asing_ditolak_bukan_diabaikan(client):
    """Salah ketik nama field TIDAK boleh lewat diam-diam."""
    salah_ketik = {**CONTOH_REQUEST, "GrLivArae": 1710}
    r = client.post("/predict", json=salah_ketik)
    assert r.status_code == 422


def test_angka_di_luar_rentang_ditolak(client):
    r = client.post("/predict", json={**CONTOH_REQUEST, "OverallQual": 99})
    assert r.status_code == 422
    assert "OverallQual" in str(r.json()["detail"])


def test_luas_negatif_ditolak(client):
    r = client.post("/predict", json={**CONTOH_REQUEST, "GrLivArea": -50})
    assert r.status_code == 422


def test_kategori_tak_dikenal_ditolak(client):
    r = client.post("/predict", json={**CONTOH_REQUEST, "Neighborhood": "Medan"})
    assert r.status_code == 422


def test_tipe_salah_ditolak(client):
    r = client.post("/predict", json={**CONTOH_REQUEST, "LotArea": "delapan ribu"})
    assert r.status_code == 422


def test_body_kosong_ditolak(client):
    assert client.post("/predict", json={}).status_code == 422


def test_batch_kosong_ditolak(client):
    assert client.post("/predict/batch", json={"houses": []}).status_code == 422


def test_batch_kebanyakan_ditolak(client):
    r = client.post("/predict/batch", json={"houses": [CONTOH_REQUEST] * (BATAS_BATCH + 1)})
    assert r.status_code == 422


def test_bentuk_error_selalu_sama(client):
    body = client.post("/predict", json={}).json()
    assert set(body) == {"request_id", "error", "detail"}


# ---------------------------------------------------------------------------
# Admin & keselarasan kontrak
# ---------------------------------------------------------------------------
def test_admin_reload_mati_tanpa_token(client):
    """Default aman: fitur berbahaya menolak, bukan terbuka."""
    r = client.post("/admin/reload")
    assert r.status_code == 503


def test_schema_selaras_dengan_kontrak_artifact(client):
    """Schema Pydantic dan kontrak di artifact wajib field yang sama persis."""
    dari_api = set(client.get("/model-info").json()["api_contract_fields"])
    assert set(HousePriceRequest.model_fields) == dari_api


def test_bsmtqual_menerima_token_tanpa_basement(client):
    r = client.post("/predict", json={**CONTOH_REQUEST, "BsmtQual": "None",
                                      "TotalBsmtSF": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0})
    assert r.status_code == 200
    assert r.json()["predicted_price_usd"] > 0


def test_kategori_di_luar_kontrak_ditolak_bukan_diabaikan(client):
    """"Po" tidak pernah muncul di ExterQual data latih, jadi harus ditolak 422.

    Kalau diterima, encoder menghasilkan one-hot nol semua dan API menjawab
    HTTP 200 dengan harga yang salah — kegagalan tanpa satu pun error.
    """
    r = client.post("/predict", json={**CONTOH_REQUEST, "ExterQual": "Po"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
def test_metrics_format_prometheus(client):
    client.post("/predict", json=CONTOH_REQUEST)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    isi = r.text
    # metrik buatan sendiri
    assert "house_price_prediksi_total" in isi
    assert "house_price_model_metrik" in isi
    # metrik sistem — didapat gratis dari prometheus_client
    assert "process_resident_memory_bytes" in isi


def test_metrics_tidak_menghitung_dirinya_sendiri(client):
    client.get("/metrics")
    isi = client.get("/metrics").text
    assert 'path="/metrics"' not in isi


def test_endpoint_drift_menjawab(client):
    body = client.get("/monitoring/drift?window=50").json()
    assert body["status"] in {"stabil", "drift_terdeteksi", "data_kurang"}
    assert body["window_diminta"] == 50


def test_drift_window_tidak_masuk_akal_ditolak(client):
    assert client.get("/monitoring/drift?window=1").status_code == 422
