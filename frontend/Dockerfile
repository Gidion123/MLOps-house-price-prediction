# =============================================================================
# Dockerfile FRONTEND — image terpisah dari backend, sengaja.
#
# Dua image, bukan satu, karena keduanya berubah dengan alasan berbeda:
# memperbaiki tata letak form tidak ada urusannya dengan model, dan melatih
# ulang model tidak ada urusannya dengan tombol. Image terpisah berarti masing-
# masing bisa dirilis sendiri tanpa menyeret yang lain.
#
# Perhatikan juga isinya: image ini TIDAK punya xgboost, scikit-learn, maupun
# artifact model. Kalau suatu hari ada yang tergoda menghitung sesuatu di sisi
# UI, kodenya tidak akan jalan di sini — dan itu memang pengamannya.
#
# Build context-nya root project (lihat docker-compose.yml), supaya `frontend`
# tetap jadi paket Python yang sama seperti saat dijalankan pytest.
# =============================================================================

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    API_URL=http://api:8000

RUN useradd --create-home --uid 10002 uiuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=uiuser:uiuser frontend/ ./frontend/

USER uiuser
EXPOSE 8501

# Streamlit menyediakan endpoint kesehatannya sendiri.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
