"""schemas.py — kontrak API dalam bentuk Pydantic. Gerbang pertama produksi.

Kenapa Pydantic, bukan ``if "GrLivArea" not in payload: ...``?

Karena validasi manual selalu bocor. Yang dicek hari ini tiga field, minggu
depan ada field keempat yang lupa dicek, dan model diam-diam menerima
``GrLivArea = -50`` lalu menjawab HTTP 200 dengan harga negatif. Pydantic
membalik bebannya: apa pun yang tidak sesuai deklarasi **ditolak sebelum
menyentuh model**, dengan pesan 422 yang menyebut field mana dan kenapa.

Tiga keputusan yang perlu dipahami:

1. **``extra="forbid"``** — field asing DITOLAK, bukan diabaikan. Kalau user
   mengirim ``"GrLivArae"`` (salah ketik), diam-diam mengabaikannya berarti
   model menerima nilai imputasi untuk ``GrLivArea`` dan menjawab dengan
   percaya diri. Silent failure paling murah yang bisa dicegah satu baris.
2. **Batas & pilihan dibaca dari ``config.yaml``**, tidak diketik ulang.
   Daftar 25 Neighborhood hidup di satu tempat: kalau model dilatih ulang
   dengan kategori baru, config yang diubah, schema ikut sendiri.
3. **Penjaga keselarasan kontrak** di akhir file: field schema dibandingkan
   dengan ``config.API_CONTRACT``. Kalau suatu hari ada yang menambah field di
   sini tapi lupa di config (atau sebaliknya), aplikasi **gagal saat start**,
   bukan saat request pertama masuk ke user.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.predict import CONTOH_REQUEST
from src.utils import config


# ---------------------------------------------------------------------------
# Pembantu: ubah aturan di config.yaml jadi tipe & batas Pydantic
# ---------------------------------------------------------------------------
def _rentang(nama: str, keterangan: str) -> Any:
    """Field numerik dengan batas min/max dari ``validation.numeric_ranges``."""
    batas = config.NUMERIC_RANGES[nama]
    return Field(..., ge=batas["min"], le=batas["max"],
                 description=f"{keterangan} (rentang sah: {batas['min']}–{batas['max']})")


def _kategori(nama: str) -> Any:
    """Tipe ``Literal`` berisi kategori sah dari ``validation.categorical_values``.

    Untuk kolom bertoken (``BsmtQual``), kata ``"None"`` dipastikan ikut. Itu
    bukan basa-basi: tanpa nilai itu, user yang rumahnya tidak punya basement
    tidak punya cara jujur mengatakannya, dan akan terpaksa mengarang kualitas
    basement yang tidak ada. Di sisi server kata itu diterjemahkan kembali
    menjadi missing value sebelum preprocessing (notebook Section 13.2).
    """
    nilai = list(config.CATEGORICAL_VALUES[nama])
    if nama in config.STRUCTURAL_CATEGORICAL and config.STRUCTURAL_API_TOKEN not in nilai:
        nilai.append(config.STRUCTURAL_API_TOKEN)
    return Literal[tuple(nilai)]  # type: ignore[valid-type]


NeighborhoodT = _kategori("Neighborhood")
MSZoningT = _kategori("MSZoning")
ExterQualT = _kategori("ExterQual")
BsmtQualT = _kategori("BsmtQual")
KitchenQualT = _kategori("KitchenQual")
HeatingQCT = _kategori("HeatingQC")
CentralAirT = _kategori("CentralAir")

BATAS_BATCH = 100


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class HousePriceRequest(BaseModel):
    """20 field yang diminta dari user — persis kontrak yang dipakai training.

    Tidak ada field "opsional yang boleh kosong" di sini. Kontraknya sengaja
    sempit supaya bisa diisi orang biasa yang melihat rumahnya, bukan 79 kolom
    yang cuma ada di dataset Kaggle.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [CONTOH_REQUEST]},
    )

    # ---- Kualitas & kondisi ----
    OverallQual: int = _rentang("OverallQual", "Kualitas material & finishing keseluruhan")
    OverallCond: int = _rentang("OverallCond", "Kondisi keseluruhan rumah")

    # ---- Luas ----
    # float, bukan int — mengikuti API_FIELD_TYPES notebook Section 13.2.
    # Luas 1710.5 sq ft adalah angka yang wajar; menolaknya hanya karena
    # "biasanya bulat" adalah penolakan yang tidak punya dasar di data.
    GrLivArea: float = _rentang("GrLivArea", "Luas lantai di atas tanah (sq ft)")
    TotalBsmtSF: float = _rentang("TotalBsmtSF", "Luas total basement (sq ft, 0 = tanpa basement)")
    LotArea: float = _rentang("LotArea", "Luas kavling (sq ft)")

    # ---- Umur bangunan ----
    YearBuilt: int = _rentang("YearBuilt", "Tahun bangunan didirikan")
    YearRemodAdd: int = _rentang("YearRemodAdd", "Tahun renovasi terakhir (= YearBuilt bila belum pernah)")

    # ---- Fasilitas ----
    GarageCars: int = _rentang("GarageCars", "Kapasitas garasi dalam jumlah mobil")
    FullBath: int = _rentang("FullBath", "Kamar mandi lengkap di atas tanah")
    HalfBath: int = _rentang("HalfBath", "Kamar mandi setengah di atas tanah")
    BsmtFullBath: int = _rentang("BsmtFullBath", "Kamar mandi lengkap di basement")
    BsmtHalfBath: int = _rentang("BsmtHalfBath", "Kamar mandi setengah di basement")
    Fireplaces: int = _rentang("Fireplaces", "Jumlah perapian")

    # ---- Kategorikal ----
    Neighborhood: NeighborhoodT = Field(..., description="Nama lingkungan di Ames, Iowa")
    MSZoning: MSZoningT = Field(..., description="Klasifikasi zonasi umum")
    ExterQual: ExterQualT = Field(..., description="Kualitas material eksterior")
    BsmtQual: BsmtQualT = Field(..., description='Tinggi/kualitas basement; "None" bila tidak punya basement')
    KitchenQual: KitchenQualT = Field(..., description="Kualitas dapur")
    HeatingQC: HeatingQCT = Field(..., description="Kualitas & kondisi pemanas")
    CentralAir: CentralAirT = Field(..., description="Punya AC sentral (Y/N)")


class BatchRequest(BaseModel):
    """Beberapa rumah sekaligus. Dibatasi supaya satu request tidak menahan worker."""

    model_config = ConfigDict(extra="forbid")

    houses: list[HousePriceRequest] = Field(
        ..., min_length=1, max_length=BATAS_BATCH,
        description=f"1–{BATAS_BATCH} rumah dalam satu panggilan",
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
# protected_namespaces=() dimatikan karena Pydantic v2 memesan awalan "model_"
# untuk dirinya sendiri, sementara di sini "model_name" adalah istilah domain
# yang benar. Mematikannya lebih jujur daripada mengganti nama field jadi
# sesuatu yang tidak dipakai orang.
_RESPONS = ConfigDict(protected_namespaces=())


class IntervalHarga(BaseModel):
    model_config = _RESPONS

    bawah: float = Field(..., description="Batas bawah perkiraan (USD)")
    atas: float = Field(..., description="Batas atas perkiraan (USD)")
    tingkat_keyakinan: float = Field(..., description="Proporsi, mis. 0.8 untuk 80%")


class PredictionResponse(BaseModel):
    """Jawaban /predict.

    Bukan cuma satu angka. Angka tunggal terbaca sebagai kepastian, padahal
    MAE model ini ~$16 ribu. Interval memaksa pembacanya melihat lebar
    ketidakpastian — dan ``request_id`` membuat tiap jawaban bisa dilacak
    balik ke barisnya di log.
    """

    model_config = _RESPONS

    request_id: str
    predicted_price_usd: float = Field(..., description="Perkiraan harga jual (USD)")
    interval: IntervalHarga
    currency: Literal["USD"] = "USD"
    model_name: str
    model_type: str
    model_trained_on_rows: int
    latency_ms: float


class BatchResponse(BaseModel):
    model_config = _RESPONS

    request_id: str
    count: int
    predictions: list[PredictionResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Jawaban /health dan /ready.

    Tiga field pertama SUDAH ADA sejak awal dan tidak diubah — Streamlit,
    healthcheck Docker, dan test lama bergantung padanya. Dua field terakhir
    ditambahkan (opsional, boleh null) supaya bisa terlihat model versi berapa
    yang sedang dilayani proses ini tanpa memanggil endpoint lain.
    """

    model_config = _RESPONS

    status: Literal["ok", "degraded"]
    model_loaded: bool
    detail: str | None = None
    model_name: str | None = None
    model_version: str | None = None


class ModelStatusResponse(BaseModel):
    """Status pergantian model — read-only, tanpa parameter, aman dibuka siapa saja.

    Endpoint status yang menerima masukan ("muat model dari path X") adalah
    lubang keamanan, bukan alat bantu. Yang ini hanya melaporkan.
    """

    model_config = _RESPONS

    sumber: str
    model_dimuat: bool
    model_name: str | None = None
    identitas_aktif: str | None = None
    identitas_champion: str | None = None
    perlu_ganti: bool
    sumber_model: dict[str, Any] | None = None
    interval_pemeriksaan_detik: int
    pemantau_aktif: bool
    catatan: str | None = None


class ModelInfoResponse(BaseModel):
    """Kartu identitas model yang sedang melayani — bahan audit, bukan hiasan."""

    model_config = _RESPONS

    model_name: str
    model_type: str
    target: str
    target_transformation: str
    training_rows: int
    created_at: str
    # Dari mana model ini dimuat: berkas joblib, atau registry versi berapa.
    # Wajib ada begitu MODEL_SOURCE=registry dipakai — dua container dengan tag
    # image yang sama bisa menyajikan versi berbeda, dan pertanyaan "prediksi
    # ini dari model mana?" harus tetap punya jawaban.
    sumber_model: dict[str, Any] | None = None
    api_contract_fields: list[str]
    engineered_features: list[str]
    leakage_features_removed: list[str]
    metrics: dict[str, float]
    metrics_note: str


class ErrorResponse(BaseModel):
    """Bentuk error yang SAMA untuk semua kegagalan.

    Client yang menghadapi tiga bentuk error berbeda akan berhenti menanganinya
    dan cuma menampilkan "terjadi kesalahan". Satu bentuk = bisa ditangani.
    """

    model_config = _RESPONS

    request_id: str
    error: str
    detail: Any | None = None


# ---------------------------------------------------------------------------
# Penjaga keselarasan: schema vs kontrak yang tersimpan di config
# ---------------------------------------------------------------------------
def _pastikan_selaras() -> None:
    di_schema = set(HousePriceRequest.model_fields)
    di_kontrak = set(config.API_CONTRACT)
    if di_schema != di_kontrak:
        raise RuntimeError(
            "KONTRAK TIDAK SELARAS antara api/schemas.py dan config.yaml. "
            f"kurang di schema={sorted(di_kontrak - di_schema)}, "
            f"berlebih di schema={sorted(di_schema - di_kontrak)}"
        )


_pastikan_selaras()
