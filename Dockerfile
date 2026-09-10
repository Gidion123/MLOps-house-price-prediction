# =============================================================================
# Dockerfile — image produksi untuk API prediksi harga rumah.
#
# Dibangun dua tahap (multi-stage). Alasannya bukan gaya-gayaan:
#
#   Tahap "builder"  memasang dependency. Butuh pip, cache, kadang compiler.
#   Tahap "runtime"  cuma menyalin hasil jadinya.
#
# Semua sampah proses instalasi tertinggal di tahap pertama dan tidak pernah
# ikut ke image akhir. Selisihnya ratusan MB — dan tiap MB yang tidak ada
# adalah satu MB yang tidak perlu dipindai keamanannya.
#
# Build & jalankan:
#     docker build -t house-price-api:1.0.0 .
#     docker run -p 8000:8000 house-price-api:1.0.0
# Atau lewat compose (disarankan, sekalian Prometheus & Grafana):
#     docker compose up -d --build
# =============================================================================

# -----------------------------------------------------------------------------
# TAHAP 1 — builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# venv terpisah supaya seluruh hasil instalasi berada di SATU folder yang
# gampang disalin utuh ke tahap runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# requirements.txt disalin SENDIRIAN dan lebih dulu dari kode.
# Docker menyimpan cache per layer: selama file ini tidak berubah, layer
# instalasi dipakai ulang. Kalau disalin bersama kode, tiap koreksi satu huruf
# di api/main.py memicu pip install ulang dari nol.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# TAHAP 2 — runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 = pustaka OpenMP. XGBoost WAJIB punya ini.
# Tanpa baris ini image berhasil dibangun, lalu mati saat start dengan
# "libgomp.so.1: cannot open shared object file". Image slim memang sengaja
# tidak membawanya, dan wheel xgboost tidak ikut memaketkannya.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_SOURCE=file \
    LOG_LEVEL=INFO

# Jalan sebagai user biasa, bukan root. Kalau suatu hari ada celah di salah satu
# dependency, penyerang mendarat sebagai user tanpa hak apa-apa — bukan sebagai
# root di dalam container.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Yang ikut masuk hanya yang dibutuhkan untuk MELAYANI request.
# Tidak ada notebooks/, tidak ada tests/, tidak ada data mentah.
COPY --chown=appuser:appuser src/    ./src/
COPY --chown=appuser:appuser api/    ./api/
COPY --chown=appuser:appuser config/ ./config/

# Artifact model IKUT dipanggang ke dalam image, bukan di-mount.
#
# Ini keputusan sadar: image jadi satu benda utuh yang isinya kode + model +
# environment, dan tag :1.0.0 menunjuk kombinasi yang persis itu. Kalau model
# di-mount dari luar, dua container dengan tag yang sama bisa melayani model
# berbeda — dan "versi mana yang menghasilkan prediksi ini?" jadi tidak
# terjawab. Harganya: ganti model = build image baru. Itu memang maksudnya.
COPY --chown=appuser:appuser models/house_price_champion.joblib \
     models/house_price_champion_metadata.json ./models/

# Folder tulis dibuat lebih dulu supaya container tidak gagal cuma karena
# logs/ belum ada.
RUN mkdir -p logs reports submissions data/raw && chown -R appuser:appuser logs reports submissions data

USER appuser
EXPOSE 8000

# Healthcheck memakai /health (liveness: prosesnya hidup?), bukan /ready.
# Kalau memakai /ready, container yang modelnya belum siap akan terus di-restart
# Docker — padahal restart tidak menyembuhkan artifact yang hilang.
# Pakai urllib, bukan curl: image slim tidak punya curl, dan memasangnya cuma
# demi healthcheck berarti menambah permukaan serangan tanpa alasan.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Satu worker. Metrik Prometheus di project ini disimpan in-process, jadi
# beberapa worker akan membuat hasil scrape melompat-lompat antar-worker.
# Menaikkan jumlah worker menuntut mode multiprocess prometheus_client lebih
# dulu — lihat catatan di src/utils/metrics.py.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
