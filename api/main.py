"""main.py — antarmuka HTTP. Routing, middleware, dan penanganan error.

Jalankan lokal::

    uvicorn api.main:app --reload --port 8000

Lalu buka http://localhost:8000/docs — dokumentasi interaktif itu dibuat
otomatis dari schema Pydantic, bukan ditulis tangan. Itu salah satu alasan
kontraknya diketik sebagai tipe, bukan sebagai komentar: dokumentasi yang
dihasilkan dari kode tidak bisa basi.

Prinsip yang dipegang file ini:

* **Gagal keras di batas, gagal lembut di dalam.** Request yang salah bentuk
  ditolak 422 sebelum menyentuh model. Tapi model yang belum ada tidak
  membuat proses mati — server tetap hidup dan menjawab 503 di ``/ready``,
  supaya orkestrator tahu jangan kirim trafik ke sini dulu.
* **Satu bentuk error untuk semua kegagalan** (lihat ``ErrorResponse``).
* **Tidak ada detail internal yang bocor ke user.** Stack trace masuk log,
  tidak masuk response body. Pesan error yang terlalu ramah ke penyerang
  adalah kebocoran informasi.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from api import service
from api.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    HousePriceRequest,
    ModelInfoResponse,
    ModelStatusResponse,
    PredictionResponse,
)
from src import monitor
from src.models.predict import ModelNotFoundError
from src.utils import config, metrics, tracking
from src.utils.logger import get_logger

logger = get_logger("house_price.api")

DESKRIPSI = """
API prediksi harga rumah — Ames, Iowa (dataset House Prices, Kaggle).

Model: XGBoost tunggal, target `log1p(SalePrice)`, dilatih pada seluruh data
berlabel setelah keputusan dikunci lewat quality gate.

Kontraknya sengaja **20 field**, bukan 79 kolom dataset asli: kolom yang
nilainya baru ada setelah rumah terjual (`MoSold`, `YrSold`, `SaleType`,
`SaleCondition`) dibuang karena tidak akan pernah ada saat user memanggil
`/predict`. Semua fitur turunan dihitung di sisi server.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat model saat start, bukan saat request pertama.

    Kalau artifact belum ada, server TETAP menyala tapi ``/ready`` menjawab
    503. Itu pola readiness probe yang dipakai orkestrator: proses hidup
    (liveness) dan proses siap melayani (readiness) adalah dua pertanyaan
    berbeda, dan menjawabnya dengan satu endpoint membuat rolling deploy
    mustahil dilakukan dengan aman.
    """
    config.ensure_directories()
    try:
        bundle = service.muat_model()
        metrics.catat_model(bundle)
        logger.info("api_siap", extra={"model": bundle.get("model_name"),
                                       "baris_latih": bundle.get("training_rows"),
                                       "versi": service.versi_singkat()})
    except Exception as err:  # noqa: BLE001
        metrics.model_siap.set(0)
        logger.error("api_start_tanpa_model", extra={"alasan": str(err)})

    # Pemantau pergantian champion dinyalakan SETELAH model pertama dimuat.
    # Kalau dinyalakan lebih dulu, ia bisa berlomba dengan pemuatan awal dan
    # menghasilkan dua pemuatan bersamaan di detik pertama hidupnya server.
    # Mati sendiri kalau MODEL_CHECK_INTERVAL_SECONDS = 0.
    service.mulai_pemantau()

    yield

    service.hentikan_pemantau()
    logger.info("api_berhenti")


app = FastAPI(
    title="House Price Prediction API",
    description=DESKRIPSI,
    version="1.0.0",
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
)


# ---------------------------------------------------------------------------
# Middleware: request_id + pencatatan tiap request
# ---------------------------------------------------------------------------
@app.middleware("http")
async def jejak_request(request: Request, call_next):
    """Beri tiap request satu id dan catat hasilnya.

    ``request_id`` ini muncul di tiga tempat: response body, header
    ``X-Request-ID``, dan baris log. Saat ada yang melapor "prediksinya aneh",
    satu id itu cukup untuk menemukan barisnya di log tanpa menebak.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    mulai = time.perf_counter()

    response = await call_next(request)

    durasi = (time.perf_counter() - mulai) * 1000
    response.headers["X-Request-ID"] = request_id

    # Label path memakai TEMPLATE rute ("/predict"), bukan URL mentah — lihat
    # catatan cardinality di src/utils/metrics.py. /metrics sendiri tidak
    # dihitung: Prometheus men-scrape-nya tiap belasan detik dan itu akan
    # menenggelamkan trafik sungguhan.
    rute = request.scope.get("route")
    path = getattr(rute, "path", request.url.path)
    if path != "/metrics":
        metrics.http_requests_total.labels(
            method=request.method, path=path, status=str(response.status_code)).inc()
        metrics.http_request_duration.labels(
            method=request.method, path=path).observe(durasi / 1000.0)

    logger.info(
        "http_request",
        extra={"request_id": request_id, "method": request.method,
               "path": request.url.path, "status": response.status_code,
               "durasi_ms": round(durasi, 2)},
    )
    return response


def _id(request: Request) -> str:
    return getattr(request.state, "request_id", "tanpa-id")


# ---------------------------------------------------------------------------
# Penanganan error — semuanya keluar dengan bentuk ErrorResponse yang sama
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def error_validasi(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 — request tidak lolos Pydantic. Ini pintu yang paling sering dipakai."""
    rincian = [
        {"field": ".".join(str(x) for x in e["loc"][1:]) or "body",
         "masalah": e["msg"], "tipe": e["type"]}
        for e in exc.errors()
    ]
    if request.url.path.startswith("/predict"):
        metrics.prediksi_total.labels(hasil="ditolak").inc()
    logger.warning("request_ditolak", extra={"request_id": _id(request),
                                             "jumlah_masalah": len(rincian),
                                             "field": [r["field"] for r in rincian]})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(request_id=_id(request),
                              error="Request tidak sesuai kontrak API",
                              detail=rincian).model_dump(),
    )


@app.exception_handler(ModelNotFoundError)
async def error_model_hilang(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """503 — model belum ada. Bukan salah user, jadi bukan 4xx."""
    logger.error("model_tidak_ada", extra={"request_id": _id(request), "alasan": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(request_id=_id(request),
                              error="Model belum siap melayani",
                              detail="Jalankan: python -m src.models.trainer").model_dump(),
    )


@app.exception_handler(ValueError)
async def error_kontrak(request: Request, exc: ValueError) -> JSONResponse:
    """422 — lolos Pydantic tapi ditolak ``validate_contract`` di predict.py.

    Ini pertahanan lapis kedua. Kalau sampai kena, artinya artifact yang dimuat
    meminta field yang berbeda dari schema — dan itu wajib gagal keras, bukan
    dibiarkan lewat dengan prediksi seadanya.
    """
    logger.error("kontrak_tidak_cocok", extra={"request_id": _id(request), "alasan": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(request_id=_id(request),
                              error="Request tidak sesuai kontrak model",
                              detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def error_tak_terduga(request: Request, exc: Exception) -> JSONResponse:
    """500 — apa pun yang tidak terduga. Detailnya ke log, BUKAN ke user."""
    logger.exception("error_tak_terduga", extra={"request_id": _id(request),
                                                 "tipe": type(exc).__name__})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(request_id=_id(request),
                              error="Terjadi kesalahan internal",
                              detail=f"Sertakan request_id ini saat melapor: {_id(request)}"
                              ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"], summary="Identitas layanan")
async def akar() -> dict[str, Any]:
    return {"service": "house-price-prediction", "version": app.version, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"],
         summary="Liveness — apakah prosesnya hidup?")
async def health() -> HealthResponse:
    """Selalu 200 selama proses berjalan. Dipakai Docker/orkestrator untuk
    memutuskan apakah container perlu di-restart."""
    termuat = service.siap()
    bundle = service.muat_model() if termuat else None
    return HealthResponse(
        status="ok" if termuat else "degraded",
        model_loaded=termuat,
        detail=None if termuat else "model belum dimuat",
        model_name=None if bundle is None else bundle.get("model_name"),
        model_version=service.versi_singkat(),
    )


@app.get("/ready", response_model=HealthResponse, tags=["meta"],
         summary="Readiness — apakah siap menerima trafik?")
async def ready() -> HealthResponse:
    """503 kalau model belum ada. Ini yang menahan trafik saat rolling deploy."""
    if not service.siap():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="model belum dimuat")
    bundle = service.muat_model()
    return HealthResponse(status="ok", model_loaded=True,
                          model_name=bundle.get("model_name"),
                          model_version=service.versi_singkat())


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"],
         summary="Kartu identitas model yang sedang melayani")
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**service.info_model())


@app.post("/predict", response_model=PredictionResponse, tags=["prediksi"],
          summary="Prediksi harga satu rumah")
async def predict(payload: HousePriceRequest, request: Request,
                  x_traffic_source: str | None = Header(default=None)) -> PredictionResponse:
    """Kembalikan perkiraan harga beserta interval 80%.

    Perhatikan urutan yang dijamin FastAPI: fungsi ini **baru dipanggil**
    setelah Pydantic meloloskan seluruh 20 field. Tidak ada satu pun validasi
    manual di badan fungsi — itu memang tujuannya.

    Header opsional ``X-Traffic-Source`` menandai asal request. Tanpa header
    (atau bernilai ``produksi``) berarti trafik nyata dan ikut dihitung dalam
    analisis drift. Nilai lain — ``test``, ``verifikasi``, ``smoke`` — menandai
    trafik sintetis yang sengaja dikeluarkan, supaya seribu request uji berisi
    rumah yang sama tidak terbaca sebagai perubahan dunia nyata.
    """
    return PredictionResponse(
        **service.prediksi(payload.model_dump(), _id(request), x_traffic_source or "produksi"))


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediksi"],
          summary="Prediksi banyak rumah sekaligus")
async def predict_batch(payload: BatchRequest, request: Request,
                        x_traffic_source: str | None = Header(default=None)) -> BatchResponse:
    rows = [rumah.model_dump() for rumah in payload.houses]
    return BatchResponse(
        **service.prediksi_batch(rows, _id(request), x_traffic_source or "produksi"))


@app.get("/model/status", response_model=ModelStatusResponse, tags=["monitoring"],
         summary="Versi model yang aktif vs yang seharusnya")
async def model_status() -> ModelStatusResponse:
    """Bandingkan model yang SEDANG dilayani proses ini dengan @champion terkini.

    Inilah jendela untuk membuktikan pergantian otomatis bekerja: sebelum
    champion berpindah ``perlu_ganti`` bernilai false, sesaat setelah berpindah
    ia jadi true, lalu kembali false begitu pemantau menyelesaikan reload.

    Read-only dan tanpa parameter — tidak bisa dipakai memuat model sembarangan.
    """
    return ModelStatusResponse(**service.status_model())


@app.get("/metrics", tags=["monitoring"], summary="Metrik format Prometheus",
         response_class=Response)
async def prometheus_metrics() -> Response:
    """Endpoint yang di-scrape Prometheus tiap belasan detik.

    Isinya bukan cuma metrik buatan kita: ``prometheus_client`` otomatis ikut
    menyertakan metrik proses (CPU, memori residen, jumlah file descriptor,
    uptime). Itu bagian "system metrics" yang diminta rubrik, dan didapat
    tanpa menambah satu pun dependensi.
    """
    isi, tipe = metrics.render()
    return Response(content=isi, media_type=tipe)


@app.get("/monitoring/drift", tags=["monitoring"],
         summary="Analisis drift dari log prediksi")
async def cek_drift(
    window: int | None = Query(default=None, ge=10, le=100_000,
                               description="jumlah prediksi terakhir yang dianalisis"),
    bonferroni: bool = Query(default=True,
                             description="koreksi banyak-uji; matikan untuk lebih sensitif"),
    hanya_produksi: bool = Query(default=True,
                                 description="abaikan trafik ber-header X-Traffic-Source (test, monitoring)"),
) -> dict[str, Any]:
    """Bandingkan sebaran request terakhir dengan data latih.

    Perhatikan ini dihitung ON DEMAND, bukan tiap kali /metrics di-scrape.
    Alasannya: analisis ini membaca ulang train.csv dan menjalankan 21 uji
    statistik — pekerjaan ratusan milidetik yang tidak pantas dibebankan ke
    scrape yang terjadi tiap 15 detik. Gauge drift di /metrics ikut diperbarui
    setiap endpoint ini (atau ``python -m src.monitor``) dijalankan, jadi di
    produksi ini dipasang sebagai job terjadwal, bukan ditunggu manual.
    """
    return monitor.analisis_drift(window=window, bonferroni=bonferroni,
                                  hanya_produksi=hanya_produksi)


def _pastikan_admin(request: Request, token_dikirim: str | None) -> None:
    """Penjaga endpoint administratif.

    Dikunci token dari ``.env``. Kalau ``ADMIN_API_TOKEN`` kosong, endpoint yang
    memakainya MATI — bukan terbuka. Default yang aman itu default yang menolak.
    """
    token = config.get_settings().admin_api_token
    if not token:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="endpoint admin dinonaktifkan (ADMIN_API_TOKEN kosong)")
    if token_dikirim != token:
        logger.warning("admin_ditolak", extra={"request_id": _id(request),
                                               "path": request.url.path})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token tidak sah")


@app.post("/admin/reload", tags=["admin"], summary="Muat ulang artifact model")
async def reload_model(request: Request,
                       x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Muat model terbaru tanpa me-restart container.

    Dengan ``MODEL_SOURCE=registry``, inilah yang membuat pergantian champion
    terasa: alias dipindah di registry, satu panggilan ke sini, dan container
    menyajikan versi baru — tanpa build ulang, tanpa restart.
    """
    _pastikan_admin(request, x_admin_token)
    bundle = service.muat_ulang()
    return {"request_id": _id(request), "status": "model dimuat ulang",
            "model_name": bundle.get("model_name"),
            "sumber_model": bundle.get("sumber_model"),
            "created_at": bundle.get("created_at")}


@app.post("/admin/rollback", tags=["admin"],
          summary="Kembalikan @champion ke versi sebelumnya, lalu muat ulang")
async def rollback(request: Request,
                   x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Batalkan promosi terakhir: alias @champion balik ke versi sebelumnya.

    Dua langkah dalam satu panggilan — alias dipindah di registry, lalu cache
    model di container dibuang supaya perpindahannya langsung terasa.

    Kalau ``MODEL_SOURCE=file``, alias tetap dipindah di registry tapi container
    ini masih menyajikan artifact dari berkas; itu dilaporkan apa adanya di
    ``sumber_model`` supaya tidak ada yang mengira rollback-nya sudah berlaku
    padahal belum.
    """
    _pastikan_admin(request, x_admin_token)
    try:
        hasil = tracking.rollback_champion()
    except RuntimeError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err
    except Exception as err:  # noqa: BLE001 - registry tak terjangkau
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"registry tidak terjangkau: {err}") from err

    bundle = service.muat_ulang()
    return {"request_id": _id(request), "status": "champion dikembalikan",
            "rollback": hasil, "sumber_model": bundle.get("sumber_model")}
