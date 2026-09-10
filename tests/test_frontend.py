"""Uji BATAS antara frontend dan backend.

Folder terpisah itu gampang; yang sulit adalah menjaganya tetap terpisah.
Pelanggaran paling umum tidak terlihat seperti pelanggaran: seseorang butuh
daftar 25 Neighborhood di form, lalu menulis ``from src.utils import config``
karena itu satu baris dan langsung jalan. Sejak saat itu frontend tidak bisa
lagi di-deploy tanpa membawa seluruh backend, dan pemisahannya tinggal nama
folder.

Test di file ini yang menahannya — bukan disiplin, bukan niat baik.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app
from frontend import kontrak
from src.utils import config

ROOT = config.PROJECT_ROOT
FRONTEND = ROOT / "frontend"
COMPOSE = ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def spec():
    with TestClient(app) as c:
        return c.get("/openapi.json").json()


# ---------------------------------------------------------------------------
# Batas yang tidak boleh dilanggar
# ---------------------------------------------------------------------------
def test_frontend_tidak_mengimpor_backend():
    """Aturan inti: frontend bicara ke backend lewat HTTP, bukan lewat import."""
    pola = re.compile(r"^\s*(?:from|import)\s+(src|api)\b", re.MULTILINE)
    pelanggar = [str(f.relative_to(ROOT)) for f in FRONTEND.rglob("*.py")
                 if pola.search(f.read_text(encoding="utf-8"))]
    assert not pelanggar, f"frontend mengimpor backend di: {pelanggar}"


def test_frontend_tidak_memasang_pustaka_ml():
    """Selama daftar ini pendek, frontend tidak MUNGKIN menghitung sendiri.

    Yang diperiksa hanya baris dependency sungguhan. Komentar sengaja dilewati —
    versi pertama test ini gagal karena membaca kalimat "frontend tidak punya
    xgboost" di komentar sebagai bukti bahwa xgboost dipasang.
    """
    baris = [b.split("#")[0].strip().lower()
             for b in (FRONTEND / "requirements.txt").read_text().splitlines()]
    paket = [b for b in baris if b]
    for terlarang in ("xgboost", "scikit-learn", "sklearn", "mlflow", "scipy", "joblib"):
        cocok = [b for b in paket if b.startswith(terlarang)]
        assert not cocok, f"{terlarang} tidak punya urusan di frontend: {cocok}"
    assert paket, "requirements frontend kosong?"


def test_frontend_tidak_menyalin_daftar_kategori():
    """Nama Neighborhood tidak boleh di-hardcode — harus datang dari /openapi.json."""
    for f in FRONTEND.rglob("*.py"):
        isi = f.read_text(encoding="utf-8")
        assert "CollgCr" not in isi, f"{f.name} menyalin kategori dari backend"
        assert "Blmngtn" not in isi, f"{f.name} menyalin kategori dari backend"


def test_alamat_backend_hanya_dari_variabel_lingkungan():
    isi = (FRONTEND / "kontrak.py").read_text()
    assert 'os.getenv("API_URL"' in isi


# ---------------------------------------------------------------------------
# Pembacaan kontrak dari backend sungguhan
# ---------------------------------------------------------------------------
def test_semua_field_kontrak_terbaca(spec):
    bidang = kontrak.bidang_dari_skema(spec)
    assert {b["nama"] for b in bidang} == set(config.API_CONTRACT)
    assert all(b["wajib"] for b in bidang)


def test_batas_numerik_ikut_terbawa(spec):
    per_nama = {b["nama"]: b for b in kontrak.bidang_dari_skema(spec)}
    for nama in config.API_CONTRACT_NUMERIC:
        b = per_nama[nama]
        assert b["min"] == config.NUMERIC_RANGES[nama]["min"]
        assert b["max"] == config.NUMERIC_RANGES[nama]["max"]
        assert b["keterangan"]


def test_pilihan_kategorikal_ikut_terbawa(spec):
    per_nama = {b["nama"]: b for b in kontrak.bidang_dari_skema(spec)}
    for nama in config.API_CONTRACT_CATEGORICAL:
        assert per_nama[nama]["pilihan"], f"{nama} tidak membawa daftar pilihan"
    # BsmtQual wajib menawarkan "None" — tanpa itu, user yang rumahnya
    # tidak punya basement terpaksa mengarang kualitas basement yang tidak ada.
    assert config.STRUCTURAL_API_TOKEN in per_nama["BsmtQual"]["pilihan"]


def test_pengelompokan_tidak_menghilangkan_field(spec):
    bidang = kontrak.bidang_dari_skema(spec)
    grup = kontrak.kelompokkan(bidang)
    assert sum(len(v) for v in grup.values()) == len(bidang)


def test_field_tak_dikenal_tetap_muncul_di_lainnya():
    """Backend menambah field baru ⇒ form ikut menampilkannya tanpa kode disentuh."""
    bidang = [{"nama": "FiturBaru", "tipe": "integer", "pilihan": None,
               "min": 0, "max": 1, "keterangan": "", "wajib": True}]
    assert kontrak.kelompokkan(bidang) == {"Lainnya": bidang}


def test_contoh_awal_mengisi_seluruh_form(spec):
    assert set(kontrak.contoh_request(spec)) == set(config.API_CONTRACT)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------
def test_streamlit_jadi_service_terpisah_yang_menunggu_api_sehat():
    compose = yaml.safe_load(COMPOSE.read_text())
    ui = compose["services"]["streamlit"]
    assert "8501:8501" in ui["ports"]
    assert "API_URL=http://api:8000" in ui["environment"]
    assert ui["depends_on"]["api"]["condition"] == "service_healthy"
    assert ui["build"]["dockerfile"] == "frontend/Dockerfile"


def test_image_frontend_dan_backend_berbeda():
    compose = yaml.safe_load(COMPOSE.read_text())
    assert compose["services"]["streamlit"]["image"] != compose["services"]["api"]["image"]


def test_container_frontend_tidak_jalan_sebagai_root():
    isi = (FRONTEND / "Dockerfile").read_text()
    assert "USER uiuser" in isi and "HEALTHCHECK" in isi
    assert isi.index("USER uiuser") < isi.index("CMD ")


def test_frontend_tidak_membawa_artifact_model():
    isi = (FRONTEND / "Dockerfile").read_text()
    assert "models/" not in isi and ".joblib" not in isi


# ---------------------------------------------------------------------------
# Integrasi: UI sungguhan menembak backend sungguhan
# ---------------------------------------------------------------------------
def test_ui_merender_form_dan_berhasil_memprediksi(monkeypatch):
    """Uji end-to-end paling menentukan: form dibangun, ditekan, backend menjawab.

    Backend dinyalakan sungguhan di port acak, lalu ``AppTest`` menjalankan
    skrip Streamlit tanpa browser. Kalau kontrak antara keduanya melenceng —
    field hilang, tipe salah, jawaban berubah bentuk — test ini yang gagal
    lebih dulu, bukan user.

    Di-skip otomatis kalau Streamlit belum terpasang, karena ia dependency
    frontend (``frontend/requirements.txt``), bukan dependency backend.
    """
    pytest.importorskip("streamlit", reason="streamlit hanya ada di environment frontend")

    import importlib
    import socket
    import threading
    import time

    import requests
    import uvicorn
    from streamlit.testing.v1 import AppTest

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        for _ in range(60):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            pytest.fail("backend tidak pernah siap")

        monkeypatch.setenv("API_URL", f"http://127.0.0.1:{port}")
        importlib.reload(kontrak)

        at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=120).run()
        assert not at.exception, f"UI gagal dirender: {at.exception}"

        # 13 numerik + 7 kategorikal = 20 field kontrak, dibangun dari /openapi.json
        assert len(at.number_input) == len(config.API_CONTRACT_NUMERIC)
        assert len(at.selectbox) == len(config.API_CONTRACT_CATEGORICAL)

        at.button[0].click().run()
        assert not at.exception, f"prediksi gagal: {at.exception}"
        assert any("berhasil" in s.value for s in at.success)
        label = {m.label for m in at.main.metric}
        assert {"Batas bawah", "Perkiraan harga", "Batas atas"} <= label
    finally:
        server.should_exit = True
        importlib.reload(kontrak)
