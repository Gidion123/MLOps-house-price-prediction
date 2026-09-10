"""Test jalur inference — yang dipakai FastAPI apa adanya."""
from __future__ import annotations

import pytest

from src.models import predict
from src.utils import config


@pytest.fixture(scope="module")
def bundle():
    return predict.load_bundle()


def test_artifact_membawa_kontrak(bundle):
    assert set(bundle["api_contract_fields"]) == set(config.API_CONTRACT)
    assert bundle["target_transformation"] == "log1p"


def test_artifact_membawa_bukti_mutu(bundle):
    for kunci in ("cv_rmse_log", "holdout_rmse_log", "holdout_mae_usd", "training_rows"):
        assert kunci in bundle, f"{kunci} tidak ada di artifact"


def test_prediksi_menghasilkan_harga_masuk_akal(bundle, request_valid):
    hasil = predict.predict_one(request_valid, bundle)
    assert 50_000 < hasil["predicted_price"] < 800_000
    assert hasil["latency_ms"] > 0


def test_request_kurang_field_ditolak(bundle, request_valid):
    request_valid.pop("GrLivArea")
    with pytest.raises(ValueError, match="kurang"):
        predict.predict_one(request_valid, bundle)


def test_request_field_berlebih_ditolak(bundle, request_valid):
    request_valid["SaleCondition"] = "Normal"      # kolom leakage
    with pytest.raises(ValueError, match="berlebih"):
        predict.predict_one(request_valid, bundle)


def test_rumah_lebih_bagus_diprediksi_lebih_mahal(bundle, request_valid):
    """Uji kewajaran arah: kualitas & luas naik -> harga naik."""
    murah = predict.predict_one(request_valid, bundle)["predicted_price"]
    mewah = dict(request_valid)
    mewah.update({"OverallQual": 10, "GrLivArea": 3000, "ExterQual": "Ex", "KitchenQual": "Ex"})
    assert predict.predict_one(mewah, bundle)["predicted_price"] > murah


def test_rumah_tanpa_basement_diterima(bundle, request_valid):
    """User harus bisa menyatakan 'tidak punya basement' lewat kontrak."""
    request_valid.update({"BsmtQual": config.STRUCTURAL_API_TOKEN, "TotalBsmtSF": 0,
                          "BsmtFullBath": 0, "BsmtHalfBath": 0})
    hasil = predict.predict_one(request_valid, bundle)
    assert hasil["predicted_price"] > 0
