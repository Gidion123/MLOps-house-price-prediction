"""data_loader.py — pos 1-3 di ban berjalan: ambil data -> periksa -> bagi.

Jalankan sebagai skrip untuk memeriksa data tanpa melatih apa pun::

    python -m src.data.data_loader

Semua keputusan di modul ini berasal dari notebook Bagian 1-3 dan
Section 13.4, dan alasannya ditulis di tempat keputusannya dipakai.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import config
from src.utils.logger import get_logger

logger = get_logger("house_price.data")


class DataValidationError(Exception):
    """Dilempar kalau data tidak lolos checkpoint.

    Sengaja exception, bukan warning: kalau training dijalankan penjadwal
    jam 2 pagi, data rusak harus MENGHENTIKAN pipeline, bukan sekadar
    mencetak peringatan yang tidak dibaca siapa-siapa (slide 25-26).
    """


def read_csv_honest(path: str | Path) -> pd.DataFrame:
    """Baca CSV tanpa membiarkan pandas menebak apa itu 'kosong'.

    Pandas punya daftar bawaan teks yang otomatis dianggap NaN, dan string
    ``"None"`` ada di daftar itu. Akibatnya kategori sah "rumah tidak punya
    lapisan batu" (864 baris) berubah jadi missing value — dan tidak ada
    error apa pun yang muncul (notebook Bagian 2.3).

    Tiga langkah, semuanya deterministik per baris sehingga aman dilakukan
    sebelum split:

    1. ``keep_default_na=False`` mematikan daftar bawaan pandas
    2. hanya sel yang benar-benar kosong yang jadi NaN
    3. token ``"NA"`` diterjemahkan sendiri, secara eksplisit
    """
    path = Path(path)
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df.replace(config.NA_TOKEN, np.nan)

    # Kolom angka yang terlanjur terbaca sebagai teks (karena berisi "NA")
    # dikembalikan ke tipe numerik. Kalau tidak, kolom seperti LotFrontage
    # akan diam-diam masuk jalur one-hot dan meledakkan matriks fitur.
    for column in df.columns:
        if df[column].dtype == object:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().sum() == df[column].notna().sum():
                df[column] = numeric

    logger.info(
        "data_dibaca",
        extra={"file": path.name, "baris": len(df), "kolom": df.shape[1],
               "missing_sel": int(df.isna().sum().sum())},
    )
    return df


def validate_raw(df: pd.DataFrame, require_target: bool = True) -> None:
    """Checkpoint kualitas data. Gagal = berhenti, bukan lanjut dengan data rusak."""
    if len(df) == 0:
        raise DataValidationError("tabel kosong")

    required = list(config.API_CONTRACT)
    if require_target:
        required = required + [config.TARGET]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise DataValidationError(f"kolom wajib tidak ada: {missing_columns}")

    if config.ID_COLUMN in df.columns and not df[config.ID_COLUMN].is_unique:
        raise DataValidationError("kolom Id tidak unik — data mungkin tergabung dua kali")

    duplicated = int(df.drop(columns=[config.ID_COLUMN], errors="ignore").duplicated().sum())
    if duplicated:
        raise DataValidationError(f"ada {duplicated} baris duplikat penuh")

    if require_target:
        invalid_target = int((df[config.TARGET] <= 0).sum())
        if invalid_target:
            raise DataValidationError(f"ada {invalid_target} baris dengan {config.TARGET} <= 0")

    # Nilai mustahil pada field kontrak: luas negatif, tahun di luar akal.
    for column, batas in config.NUMERIC_RANGES.items():
        if column not in df.columns:
            continue
        di_luar = int(((df[column] < batas["min"]) | (df[column] > batas["max"])).sum())
        if di_luar:
            raise DataValidationError(
                f"{column}: {di_luar} nilai di luar rentang wajar "
                f"[{batas['min']}, {batas['max']}]"
            )

    logger.info(
        "validasi_lolos",
        extra={"baris": len(df), "duplikat": duplicated,
               "field_kontrak_dicek": len(config.API_CONTRACT)},
    )


def contract_signature(df: pd.DataFrame) -> pd.Series:
    """Sidik jari sebuah baris dilihat dari mata model: field kontrak jadi teks.

    Dipakai untuk mendeteksi 'kembar-fitur' — dua rumah yang berbeda di file
    tetapi identik di kolom yang benar-benar dipakai model.
    """
    return df[config.API_CONTRACT].astype(str).agg("|".join, axis=1)


def purify_holdout(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Buang baris holdout yang sidik jarinya sudah ada di train.

    Kalau dua rumah identik di 20 field kontrak dan satu ada di train sementara
    kembarannya di holdout, model sudah 'melihat jawabannya' saat latihan —
    skor holdout jadi optimistis palsu (notebook Section 13.4).

    Yang dibuang HANYA baris holdout. Train tidak disentuh, karena train justru
    butuh melihat variasi harga di rumah identik: itu yang mengajari model bahwa
    ada noise yang memang tidak bisa dijelaskan.
    """
    train_signatures = set(contract_signature(train))
    contaminated = contract_signature(holdout).isin(train_signatures)

    if contaminated.any():
        logger.warning(
            "holdout_tercemar",
            extra={"jumlah": int(contaminated.sum()),
                   "id": holdout.loc[contaminated, config.ID_COLUMN].tolist()},
        )

    clean = holdout.loc[~contaminated].copy()
    assert not contract_signature(clean).isin(train_signatures).any()
    return clean


def load_train_holdout(
    path: str | Path | None = None,
    dengan_statistik: bool = False,
):
    """Muat train.csv, validasi, bagi jadi train/holdout, lalu purifikasi holdout.

    Split memakai ``random_state`` dari config supaya pembagiannya identik
    di laptop siapa pun dan di dalam container.

    ``dengan_statistik=True`` menambahkan dict berisi jumlah baris di tiap
    tahap. Berapa baris yang dibuang purifikasi adalah bagian dari bukti
    evaluasi — kalau angkanya cuma lewat di log, ia tidak ikut ke metadata
    artifact dan tidak bisa diperiksa lagi tiga bulan kemudian. Bawaannya
    tetap 2-tuple supaya pemanggil lama tidak perlu diubah.
    """
    df = read_csv_honest(path or config.TRAIN_CSV)
    validate_raw(df)

    train, holdout = train_test_split(
        df, test_size=config.HOLDOUT_SIZE, random_state=config.RANDOM_STATE, shuffle=True
    )
    if len(train) + len(holdout) != len(df):
        raise DataValidationError("ada baris hilang saat split")

    holdout_clean = purify_holdout(train, holdout)

    statistik = {
        "baris_berlabel": int(len(df)),
        "train_rows": int(len(train)),
        "holdout_rows_awal": int(len(holdout)),
        "holdout_rows": int(len(holdout_clean)),
        "holdout_excluded_rows": int(len(holdout) - len(holdout_clean)),
    }
    logger.info("split_selesai", extra=statistik)

    if dengan_statistik:
        return train, holdout_clean, statistik
    return train, holdout_clean


def load_scoring_data(path: str | Path | None = None) -> pd.DataFrame:
    """Muat test.csv (tanpa target) untuk keperluan submission."""
    df = read_csv_honest(path or config.TEST_CSV)
    validate_raw(df, require_target=False)
    return df


def main() -> None:
    config.ensure_directories()
    train, holdout = load_train_holdout()
    print(f"train  : {len(train)} baris")
    print(f"holdout: {len(holdout)} baris (sudah dipurifikasi)")
    print(f"kontrak: {len(config.API_CONTRACT)} field")


if __name__ == "__main__":
    main()
