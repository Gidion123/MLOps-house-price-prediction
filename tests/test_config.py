"""Test integritas kontrak API.

Test di sini murah tapi berharga: dia menangkap kesalahan konfigurasi yang
kalau lolos akan menghasilkan model yang tidak bisa di-deploy — persis
kesalahan yang pernah terjadi di notebook (model 79 fitur vs kontrak 20 field).
"""
from __future__ import annotations

from src.utils import config


def test_kontrak_tidak_mengandung_kolom_leakage():
    assert not set(config.API_CONTRACT) & set(config.LEAKAGE_COLUMNS)


def test_kolom_model_hanya_berasal_dari_kontrak_dan_turunan():
    diminta = [c for c in config.MODEL_COLUMNS if c not in config.ENGINEERED_FEATURES]
    assert set(diminta) == set(config.API_CONTRACT)


def test_tidak_ada_field_ganda_di_kontrak():
    assert len(config.API_CONTRACT) == len(set(config.API_CONTRACT))


def test_setiap_field_punya_aturan_validasi():
    for field in config.API_CONTRACT_NUMERIC:
        assert field in config.NUMERIC_RANGES, f"{field} belum punya rentang validasi"
    for field in config.API_CONTRACT_CATEGORICAL:
        assert field in config.CATEGORICAL_VALUES, f"{field} belum punya daftar nilai sah"


def test_kategori_bertoken_punya_nilai_pengganti_di_daftar_sah():
    """BsmtQual harus punya token "None" supaya user bisa bilang tidak punya basement."""
    for field in config.STRUCTURAL_CATEGORICAL:
        assert config.STRUCTURAL_API_TOKEN in config.CATEGORICAL_VALUES[field]


def test_kategori_sah_tidak_menjanjikan_nilai_di_luar_data_latih():
    """Kategori yang tak pernah dilihat encoder jadi one-hot nol semua — tanpa error.

    Menawarkannya di kontrak sama dengan menjanjikan jawaban yang tidak bisa
    diberikan model. Daftar kategori WAJIB berasal dari train.csv, bukan dari
    ingatan tentang skala kualitas Ames yang "seharusnya" punya Po.
    """
    from src.data.data_loader import read_csv_honest

    train = read_csv_honest(config.TRAIN_CSV)
    for field in config.API_CONTRACT_CATEGORICAL:
        di_data = set(train[field].dropna().astype(str))
        di_kontrak = set(config.CATEGORICAL_VALUES[field]) - {config.STRUCTURAL_API_TOKEN}
        assert di_kontrak <= di_data, (
            f"{field}: {sorted(di_kontrak - di_data)} ditawarkan kontrak "
            "tapi tidak pernah muncul di data latih"
        )
        assert di_data <= di_kontrak, (
            f"{field}: {sorted(di_data - di_kontrak)} ada di data latih "
            "tapi tidak bisa dikirim user"
        )


def test_ambang_gate_masuk_akal():
    assert 0 < config.GATE_MAX_CV_RMSE_LOG < 1
    assert config.GATE_BASELINE_RMSE_LOG <= config.GATE_MAX_CV_RMSE_LOG
