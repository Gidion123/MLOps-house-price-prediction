"""data_preprocessor.py — pipeline preprocessing, satu objek satu artifact.

Prinsip yang dijaga modul ini (slide 29-30):

    Jalur kode training dan serving berbeda, kecepatannya berbeda, datanya
    berbeda — tetapi **preprocessing-nya harus persis sama**.

Karena itu semua langkah transformasi hidup di dalam satu ``Pipeline`` yang
disimpan sebagai satu artifact. Saat serving kita cuma memanggil ``.transform()``
dan tidak pernah menghitung ulang apa pun. Itu yang menutup *training-serving
skew*, kegagalan paling mahal di ML yang tidak memunculkan satu pun error.

Bonus dari pola yang sama: karena ``fit`` hanya dipanggil di dalam fold
cross-validation, *preprocessing leakage* juga ikut mustahil terjadi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.utils import config
from src.utils.logger import get_logger

logger = get_logger("house_price.preprocess")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan 3 fitur turunan — HANYA dari field yang ada di kontrak API.

    Ini batasan yang gampang dilanggar tanpa sadar. ``TotalSF`` versi umum
    dihitung dari ``TotalBsmtSF + 1stFlrSF + 2ndFlrSF``, padahal dua kolom
    terakhir tidak diminta dari user. Kalau dipakai, komponen yang hilang
    terisi 0 dan hasilnya diam-diam salah.

    Versi hemat di bawah memakai ``GrLivArea`` yang ada di kontrak. Di notebook
    Bagian 4.15 terbukti korelasi dua versi itu 0.9979 — manfaatnya sama,
    tanpa menambah field yang harus diisi user.
    """
    out = df.copy()

    def col(name: str):
        return out[name].fillna(0) if name in out.columns else 0

    out["TotalSF"] = col("TotalBsmtSF") + col("GrLivArea")
    out["TotalBath"] = (
        col("FullBath") + 0.5 * col("HalfBath")
        + col("BsmtFullBath") + 0.5 * col("BsmtHalfBath")
    )
    out["QualxArea"] = col("OverallQual") * col("GrLivArea")
    return out


def token_kontrak_ke_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Terjemahkan token API ``"None"`` kembali menjadi missing value.

    Ini padanan persis ``add_api_safe_fe`` di notebook Section 13.2, dan
    arah terjemahannya penting.

    Di ``train.csv``, ``BsmtQual`` kosong berarti "rumah ini tidak punya
    basement". Lewat HTTP, user tidak punya cara mengirim "kosong" — ia butuh
    sebuah kata, dan kontrak memilih kata ``"None"``. Kata itu HARUS dilepas
    lagi sebelum menyentuh preprocessing, supaya baris yang masuk lewat API
    diperlakukan sama persis dengan baris yang sama di data latih.

    Kalau langkah ini dilewat, ``"None"`` menjadi kategori yang tidak pernah
    dilihat encoder. ``handle_unknown="ignore"`` membuatnya jadi one-hot nol
    semua — prediksinya salah, HTTP-nya tetap 200, dan tidak ada satu pun
    error yang muncul.
    """
    out = df.copy()
    for column in config.STRUCTURAL_CATEGORICAL:
        if column in out.columns:
            # Penugasan bermasker, bukan .replace(): sejak pandas 2.2 `replace`
            # diam-diam menurunkan dtype kolom dan mengeluarkan FutureWarning.
            kolom = out[column]
            out[column] = kolom.mask(kolom.astype("object") == config.STRUCTURAL_API_TOKEN)
    return out


def nan_ke_token_kontrak(df: pd.DataFrame) -> pd.DataFrame:
    """Arah sebaliknya: NaN -> ``"None"``, untuk membandingkan data latih
    dengan request yang masuk.

    Dipakai monitoring drift. Distribusi referensi dibaca dari ``train.csv``
    (``BsmtQual`` = NaN), sedangkan log request berisi kata ``"None"``. Tanpa
    penyamaan ini, dua penamaan untuk hal yang sama akan terbaca sebagai
    perubahan distribusi — alarm yang benar secara statistik tapi salah
    secara makna.
    """
    out = df.copy()
    for column in config.STRUCTURAL_CATEGORICAL:
        if column in out.columns:
            out[column] = out[column].fillna(config.STRUCTURAL_API_TOKEN)
    return out


def prepare_features(
    df: pd.DataFrame,
    normalisasi_token: bool = True,
    structural: bool = True,
) -> pd.DataFrame:
    """Semua transformasi DETERMINISTIK per baris, lalu pilih kolom model.

    Deterministik = hasilnya untuk satu baris tidak bergantung pada baris lain.
    Karena itu langkah-langkah ini aman berada di luar ``fit`` — dan justru
    harus berada di dalam Pipeline supaya ikut tersimpan di artifact.

    Urutannya mengikuti ``add_api_safe_fe`` notebook Section 13.2: token
    kontrak dilepas lebih dulu, baru fitur turunan dihitung.
    """
    out = token_kontrak_ke_nan(df) if normalisasi_token else df
    out = add_engineered_features(out)
    hasil = out[config.MODEL_COLUMNS]

    # Dicatat hanya untuk batch (training, CV, submission), bukan untuk request
    # tunggal. Alasannya sederhana: satu request sudah menghasilkan dua baris log
    # ("http_request" dan "prediksi"); menambah baris ketiga yang isinya selalu
    # "1 baris masuk, 1 baris keluar" cuma menggandakan biaya penyimpanan log
    # tanpa menambah satu pun informasi baru.
    if len(hasil) > 1:
        logger.info("preprocessing", extra={
            "baris": len(hasil),
            "kolom_masuk": int(df.shape[1]),
            "kolom_keluar": len(config.MODEL_COLUMNS),
            "missing_tersisa": int(hasil.isna().sum().sum()),
            "normalisasi_token": normalisasi_token,
        })
    return hasil


def build_preprocessor(normalisasi_token: bool = True) -> Pipeline:
    """Pipeline preprocessing lengkap: DataFrame mentah -> matriks siap model.

    Dua tahap:

    1. ``prepare``  — token kontrak -> NaN, lalu fitur turunan (deterministik)
    2. ``encode``   — imputasi statistik + one-hot (dipelajari dari training)

    Tahap 2 yang ``fit``-nya harus dijaga: median dan modus dihitung dari data,
    jadi kalau dihitung dari seluruh dataset sebelum split, informasi holdout
    menetes ke train.
    """
    numeric = config.API_CONTRACT_NUMERIC + config.ENGINEERED_FEATURES
    categorical = config.API_CONTRACT_CATEGORICAL

    encode = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy=config.NUMERIC_IMPUTE_STRATEGY), numeric),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy=config.CATEGORICAL_IMPUTE_STRATEGY)),
                    # handle_unknown="ignore": kategori yang belum pernah dilihat
                    # tidak membuat API crash. Pertahanan lapis kedua setelah
                    # Enum Pydantic yang menolaknya lebih dulu dengan 422.
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )

    logger.info("preprocessor_dibangun", extra={
        "n_numerik": len(numeric), "n_kategorikal": len(categorical),
        "normalisasi_token": normalisasi_token,
    })

    return Pipeline([
        ("prepare", FunctionTransformer(
            prepare_features, validate=False,
            kw_args={"normalisasi_token": normalisasi_token},
        )),
        ("encode", encode),
    ])


def transform_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """SalePrice -> log1p(SalePrice). Target miring parah (skew 1.74)."""
    return np.log1p(np.asarray(y, dtype=float))


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """log1p -> USD. Dipanggil sebelum angka dikirim ke user.

    Kalau langkah ini terlewat, API tetap menjawab HTTP 200 — dengan harga
    rumah sekitar $12. Nol error, nol alarm.
    """
    return np.maximum(np.expm1(np.asarray(y_log, dtype=float)), 0.0)
