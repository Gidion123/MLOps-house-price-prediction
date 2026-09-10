"""Test pipeline preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data_preprocessor import (
    add_engineered_features,
    nan_ke_token_kontrak,
    token_kontrak_ke_nan,
    build_preprocessor,
    inverse_transform_target,
    prepare_features,
    transform_target,
)
from src.utils import config


def test_fitur_turunan_dihitung_benar():
    df = pd.DataFrame([{
        "TotalBsmtSF": 800, "GrLivArea": 1500, "OverallQual": 7,
        "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
    }])
    hasil = add_engineered_features(df)
    assert hasil["TotalSF"].iloc[0] == 2300           # 800 + 1500
    assert hasil["TotalBath"].iloc[0] == 3.5          # 2 + 0.5 + 1 + 0
    assert hasil["QualxArea"].iloc[0] == 10500        # 7 * 1500


def test_fitur_turunan_hanya_memakai_field_kontrak():
    """Kalau turunan butuh kolom di luar kontrak, API diam-diam ikut membengkak."""
    df = pd.DataFrame([{k: 1 for k in config.API_CONTRACT}])
    hasil = add_engineered_features(df)
    assert set(config.ENGINEERED_FEATURES).issubset(hasil.columns)


def test_token_kontrak_dikembalikan_jadi_missing_value():
    """Token "None" dari API harus jadi NaN lagi sebelum menyentuh preprocessing."""
    df = pd.DataFrame({"BsmtQual": ["Gd", config.STRUCTURAL_API_TOKEN, "TA"]})
    hasil = token_kontrak_ke_nan(df)
    assert hasil["BsmtQual"].isna().tolist() == [False, True, False]
    assert hasil["BsmtQual"].dropna().tolist() == ["Gd", "TA"]


def test_penyamaan_token_bolak_balik_konsisten():
    """Arah sebaliknya dipakai monitoring: NaN di data latih == "None" di request."""
    df = pd.DataFrame({"BsmtQual": ["Gd", None, "TA"]})
    assert nan_ke_token_kontrak(df)["BsmtQual"].tolist() == [
        "Gd", config.STRUCTURAL_API_TOKEN, "TA"
    ]
    bolak_balik = token_kontrak_ke_nan(nan_ke_token_kontrak(df))
    assert bolak_balik["BsmtQual"].isna().tolist() == df["BsmtQual"].isna().tolist()


def test_request_tanpa_basement_tidak_jadi_one_hot_nol_semua(data_split):
    """Kegagalan diam yang paling mahal: "None" dianggap kategori tak dikenal.

    Kalau itu terjadi, seluruh kolom one-hot BsmtQual bernilai 0 dan API tetap
    menjawab HTTP 200 dengan angka yang salah. Test ini membandingkan baris
    bertoken dengan baris ber-NaN: keduanya harus menghasilkan matriks identik.
    """
    train, _ = data_split
    pre = build_preprocessor()
    pre.fit(train)

    baris = train.iloc[[0]].copy()
    baris["BsmtQual"] = config.STRUCTURAL_API_TOKEN
    lewat_token = pre.transform(baris)

    baris_nan = train.iloc[[0]].copy()
    baris_nan["BsmtQual"] = np.nan
    lewat_nan = pre.transform(baris_nan)

    assert np.allclose(lewat_token.toarray() if hasattr(lewat_token, "toarray") else lewat_token,
                       lewat_nan.toarray() if hasattr(lewat_nan, "toarray") else lewat_nan)


def test_prepare_menghasilkan_kolom_model_persis(data_split):
    train, _ = data_split
    hasil = prepare_features(train)
    assert list(hasil.columns) == config.MODEL_COLUMNS


def test_preprocessor_tidak_menghasilkan_nan(data_split):
    train, holdout = data_split
    pre = build_preprocessor()
    X = pre.fit_transform(train)
    Xh = pre.transform(holdout)
    assert not np.isnan(X).any()
    assert not np.isnan(Xh).any()
    assert X.shape[1] == Xh.shape[1]


def test_kategori_tak_dikenal_tidak_bikin_crash(data_split):
    """handle_unknown='ignore' — pertahanan lapis kedua setelah Enum Pydantic."""
    train, _ = data_split
    pre = build_preprocessor()
    pre.fit(train)
    aneh = train.head(1).copy()
    aneh.loc[aneh.index[0], "Neighborhood"] = "Kemang"   # tidak ada di Ames
    hasil = pre.transform(aneh)
    assert hasil.shape[0] == 1


def test_transformasi_target_bolak_balik():
    harga = np.array([34900.0, 163000.0, 755000.0])
    kembali = inverse_transform_target(transform_target(harga))
    np.testing.assert_allclose(kembali, harga, rtol=1e-9)


def test_inverse_tidak_pernah_negatif():
    assert (inverse_transform_target(np.array([-5.0, 0.0])) >= 0).all()
