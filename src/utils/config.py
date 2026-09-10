"""config.py — manajemen konfigurasi terpusat.

Dua sumber, dua tanggung jawab yang berbeda:

1. ``config/config.yaml``  -> keputusan ML yang boleh dibaca siapa saja
                              (kontrak API, hyperparameter, ambang gate).
2. ``.env``                -> rahasia & pengaturan lingkungan
                              (URI MLflow, port, level log). TIDAK masuk Git.

Kenapa dipisah begitu? Karena keputusan ML perlu di-review lewat pull request,
sementara rahasia justru tidak boleh terlihat sama sekali (slide 28).

Semua modul lain mengimpor dari sini, tidak ada yang membaca YAML sendiri —
supaya cuma ada satu sumber kebenaran.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path proyek. Dihitung dari lokasi file ini, bukan dari cwd, supaya kode tetap
# benar dijalankan dari root project, dari notebooks/, maupun dari dalam Docker.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
MODEL_ARTIFACT = MODELS_DIR / "house_price_champion.joblib"
MODEL_METADATA = MODELS_DIR / "house_price_champion_metadata.json"
PREDICTION_LOG = LOGS_DIR / "predictions.log"


class Settings(BaseSettings):
    """Pengaturan lingkungan, dibaca dari .env atau environment variable.

    Pakai pydantic-settings supaya nilainya divalidasi tipenya saat startup,
    bukan meledak di tengah request karena PORT ternyata berisi teks.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "house-price-prediction"
    mlflow_model_name: str = "house-price-regressor"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # "registry" | "file" | "auto" — dipakai api/ saat memuat model.
    model_source: str = "auto"

    # Berapa detik sekali API memeriksa apakah champion sudah berganti.
    # 0 = pemantauan MATI (perilaku lama: model hanya berganti lewat
    # /admin/reload atau restart). Sengaja 0 sebagai bawaan supaya menyalakan
    # pergantian otomatis adalah keputusan sadar, bukan efek samping —
    # dan supaya test berjalan deterministik tanpa thread latar.
    model_check_interval_seconds: int = 0

    # Token untuk endpoint administratif (/admin/reload). Kosong = endpoint-nya
    # MATI. Defaultnya sengaja kosong: fitur berbahaya harus dinyalakan dengan
    # sengaja, bukan aktif duluan lalu diamankan belakangan (slide 28).
    admin_api_token: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings di-cache: file .env cukup dibaca sekali per proses."""
    return Settings()


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Muat config.yaml. Di-cache supaya tidak dibaca ulang tiap dipanggil."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"config.yaml tidak ditemukan di {CONFIG_FILE}. "
            "Jalankan dari root project atau periksa struktur folder."
        )
    with CONFIG_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = load_config()

# ---------------------------------------------------------------------------
# Konstanta yang sering dipakai, diangkat ke level modul supaya pemanggilnya
# rapi: `from src.utils.config import RANDOM_STATE` — bukan menggali dict.
# ---------------------------------------------------------------------------
RANDOM_STATE: int = _CFG["project"]["random_state"]
HOLDOUT_SIZE: float = _CFG["project"]["holdout_size"]

TARGET: str = _CFG["data"]["target"]
ID_COLUMN: str = _CFG["data"]["id_column"]
NA_TOKEN: str = _CFG["data"]["na_token"]

LEAKAGE_COLUMNS: list[str] = list(_CFG["leakage_columns"])

API_CONTRACT_NUMERIC: list[str] = list(_CFG["api_contract"]["numeric"])
API_CONTRACT_CATEGORICAL: list[str] = list(_CFG["api_contract"]["categorical"])
API_CONTRACT: list[str] = API_CONTRACT_NUMERIC + API_CONTRACT_CATEGORICAL

# Tipe tiap field numerik kontrak (notebook Section 13.2). Dipakai Pydantic
# untuk memilih int vs float, dan ikut masuk ke contract_hash.
API_CONTRACT_INT: list[str] = list(_CFG["api_contract"]["integer_fields"])
API_CONTRACT_FLOAT: list[str] = list(_CFG["api_contract"]["float_fields"])
API_FIELD_TYPES: dict[str, str] = {
    **{c: "int" for c in API_CONTRACT_INT},
    **{c: "float" for c in API_CONTRACT_FLOAT},
}

ENGINEERED_FEATURES: list[str] = list(_CFG["engineered_features"])
MODEL_COLUMNS: list[str] = API_CONTRACT + ENGINEERED_FEATURES

STRUCTURAL_CATEGORICAL: list[str] = list(_CFG["imputation"]["structural_categorical"])
# Kata yang dikirim user untuk "fitur ini memang tidak ada". Di sisi server ia
# dikembalikan menjadi NaN sebelum preprocessing — lihat data_preprocessor.
STRUCTURAL_API_TOKEN: str = _CFG["imputation"]["structural_api_token"]
NUMERIC_IMPUTE_STRATEGY: str = _CFG["imputation"]["numeric_strategy"]
CATEGORICAL_IMPUTE_STRATEGY: str = _CFG["imputation"]["categorical_strategy"]

NUMERIC_RANGES: dict[str, dict[str, float]] = _CFG["validation"]["numeric_ranges"]
CATEGORICAL_VALUES: dict[str, list[str]] = _CFG["validation"]["categorical_values"]

MODEL_NAME: str = _CFG["model"]["name"]
MODEL_VERSION: str = str(_CFG["model"]["version"])
TARGET_TRANSFORMATION: str = _CFG["model"]["target_transformation"]
INVERSE_TRANSFORMATION: str = _CFG["model"]["inverse_transformation"]
HYPERPARAMETERS: dict[str, Any] = dict(_CFG["model"]["hyperparameters"])
CV_FOLDS: int = _CFG["training"]["cv_folds"]

GATE_MAX_CV_RMSE_LOG: float = _CFG["quality_gate"]["max_cv_rmse_log"]
GATE_BASELINE_RMSE_LOG: float = _CFG["quality_gate"]["baseline_rmse_log"]
GATE_MAX_HOLDOUT_GAP_RATIO: float = _CFG["quality_gate"]["max_holdout_gap_ratio"]

DRIFT_P_VALUE_THRESHOLD: float = _CFG["monitoring"]["drift_p_value_threshold"]
MONITOR_WINDOW_SIZE: int = _CFG["monitoring"]["window_size"]

PROMOSI_MARGIN_STD_FAKTOR: float = _CFG.get("registry", {}).get("margin_std_faktor", 0.0)
PROMOSI_WAJIB_KONTRAK_SAMA: bool = _CFG.get("registry", {}).get("wajib_kontrak_sama", True)


def ensure_directories() -> None:
    """Buat folder hasil kerja kalau belum ada.

    Dipanggil di awal training & serving supaya tidak gagal cuma karena
    folder logs/ belum ada di container yang baru dibangun.
    """
    for folder in (DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR, SUBMISSIONS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
