"""Uji berkas konfigurasi infrastruktur — config as code, jadi ikut diuji.

Kesalahan di file YAML/JSON punya sifat menyebalkan: tidak ada yang menegur
saat ditulis, dan baru ketahuan setelah `docker compose up` menghasilkan
dashboard kosong atau alert yang tidak pernah bunyi. Nama metrik yang salah
ketik satu huruf terlihat persis sama dengan metrik yang nilainya nol.

Uji di sini menutup celah itu: tiap nama metrik yang disebut dashboard dan
alert dicocokkan dengan metrik yang BENAR-BENAR diekspor aplikasi.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from src.utils import config, metrics

ROOT = config.PROJECT_ROOT
COMPOSE = ROOT / "docker-compose.yml"
PROMETHEUS = ROOT / "monitoring" / "prometheus.yml"
ALERTS = ROOT / "monitoring" / "alerts.yml"
DASHBOARD = ROOT / "monitoring" / "grafana" / "dashboards" / "house-price.json"
DATASOURCE = ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "datasource.yml"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def metrik_terekspor() -> str:
    """Isi /metrics setelah aplikasi benar-benar melayani satu request.

    Pemanasan ini bukan formalitas. ``prometheus_client`` baru menuliskan baris
    sampel untuk metrik BERLABEL setelah ada kombinasi label yang sungguhan
    diamati. Pada proses yang baru start, ``house_price_http_request_duration_
    seconds_bucket`` belum ada satu baris pun — yang ada cuma HELP dan TYPE.
    Membandingkan dashboard dengan /metrics yang dingin akan melaporkan metrik
    yang sebenarnya sehat sebagai "tidak ada".
    """
    from fastapi.testclient import TestClient

    from api.main import app
    from src.models.predict import CONTOH_REQUEST

    with TestClient(app) as c:
        c.headers["X-Traffic-Source"] = "test"
        c.get("/health")
        c.post("/predict", json=CONTOH_REQUEST)
        return c.get("/metrics").text


def _nama_metrik(teks: str) -> set[str]:
    """Semua nama metrik house_price_* yang disebut di sebuah file."""
    return set(re.findall(r"house_price_[a-z_]+", teks))


# ---------------------------------------------------------------------------
# Yang paling penting: nama metrik harus benar-benar ada
# ---------------------------------------------------------------------------
def test_metrik_di_dashboard_memang_diekspor(metrik_terekspor):
    disebut = _nama_metrik(DASHBOARD.read_text())
    assert disebut, "dashboard tidak menyebut satu pun metrik — pasti ada yang salah"
    hilang = sorted(n for n in disebut if n not in metrik_terekspor)
    assert not hilang, f"dashboard menunjuk metrik yang tidak ada: {hilang}"


def test_metrik_di_alert_memang_diekspor(metrik_terekspor):
    disebut = _nama_metrik(ALERTS.read_text())
    hilang = sorted(n for n in disebut if n not in metrik_terekspor)
    assert not hilang, f"alert menunjuk metrik yang tidak ada: {hilang}"


# ---------------------------------------------------------------------------
# Sambungan antar-service
# ---------------------------------------------------------------------------
def test_prometheus_menembak_service_yang_benar():
    prom = yaml.safe_load(PROMETHEUS.read_text())
    job = next(j for j in prom["scrape_configs"] if j["job_name"] == "house-price-api")
    assert job["metrics_path"] == "/metrics"
    # Harus nama service compose, bukan localhost. Di dalam jaringan container,
    # "localhost" berarti container Prometheus itu sendiri.
    assert job["static_configs"][0]["targets"] == ["api:8000"]


def test_target_scrape_cocok_dengan_port_compose():
    compose = yaml.safe_load(COMPOSE.read_text())
    assert "api" in compose["services"]
    assert "8000:8000" in compose["services"]["api"]["ports"]


def test_uid_datasource_cocok_dengan_dashboard():
    """uid dikunci manual; kalau tidak cocok, semua panel jadi 'datasource not found'."""
    ds = yaml.safe_load(DATASOURCE.read_text())["datasources"][0]
    dash = json.loads(DASHBOARD.read_text())
    uid_panel = {p["datasource"]["uid"] for p in dash["panels"]}
    assert uid_panel == {ds["uid"]} == {"prometheus"}
    assert ds["url"] == "http://prometheus:9090"


def test_semua_service_ada_dan_terhubung():
    compose = yaml.safe_load(COMPOSE.read_text())
    assert set(compose["services"]) == {"api", "streamlit", "prometheus", "grafana"}
    for nama in compose["services"]:
        assert "mlops" in compose["services"][nama]["networks"]


def test_prometheus_memuat_berkas_alert():
    prom = yaml.safe_load(PROMETHEUS.read_text())
    assert "/etc/prometheus/alerts.yml" in prom["rule_files"]
    compose = yaml.safe_load(COMPOSE.read_text())
    mounts = compose["services"]["prometheus"]["volumes"]
    assert any("alerts.yml:/etc/prometheus/alerts.yml" in m for m in mounts)


# ---------------------------------------------------------------------------
# Alert harus bisa ditindaklanjuti
# ---------------------------------------------------------------------------
def test_setiap_alert_punya_jeda_dan_tindakan():
    """Alert tanpa `for:` akan bunyi karena satu scrape meleset saat deploy.
    Alert tanpa `tindakan` tidak bisa dijawab jam 2 pagi."""
    grup = yaml.safe_load(ALERTS.read_text())["groups"]
    aturan = [r for g in grup for r in g["rules"]]
    assert len(aturan) >= 5
    for r in aturan:
        assert r.get("for"), f"{r['alert']} tidak punya klausa for"
        assert r["annotations"].get("tindakan"), f"{r['alert']} tidak menyebut tindakan"
        assert r["labels"]["keparahan"] in {"kritis", "peringatan"}


# ---------------------------------------------------------------------------
# Keamanan & kebenaran image
# ---------------------------------------------------------------------------
def test_env_tidak_ikut_masuk_image():
    """Rahasia yang terpanggang di layer image bisa dibongkar siapa pun."""
    isi = DOCKERIGNORE.read_text()
    assert "\n.env\n" in isi and ".env.*" in isi


def test_data_dan_notebook_tidak_ikut_masuk_image():
    isi = DOCKERIGNORE.read_text()
    for pola in ("data/", "notebooks/", "mlruns/", "tests/"):
        assert pola in isi


def test_artifact_model_justru_harus_ikut():
    """Kebalikannya: models/ TIDAK boleh diabaikan, image butuh artifact-nya."""
    baris = [b.strip() for b in DOCKERIGNORE.read_text().splitlines()
             if b.strip() and not b.strip().startswith("#")]
    assert "models/" not in baris
    assert "house_price_champion.joblib" in DOCKERFILE.read_text()


def test_dockerfile_memasang_libgomp():
    """Tanpa libgomp1, image berhasil dibangun lalu mati saat import xgboost."""
    assert "libgomp1" in DOCKERFILE.read_text()


def test_container_tidak_jalan_sebagai_root():
    isi = DOCKERFILE.read_text()
    assert "USER appuser" in isi
    assert isi.index("USER appuser") < isi.index("CMD ")


def test_dockerfile_multistage_dan_punya_healthcheck():
    isi = DOCKERFILE.read_text()
    assert isi.count("FROM ") >= 2 and "AS builder" in isi
    assert "HEALTHCHECK" in isi
    # /health (liveness), bukan /ready — restart tidak menyembuhkan artifact hilang.
    assert "/health'" in isi and "/ready'" not in isi


def test_data_latih_dipasang_readonly():
    """Analisis drift butuh train.csv, tapi container tidak boleh bisa menulisinya."""
    compose = yaml.safe_load(COMPOSE.read_text())
    mounts = compose["services"]["api"]["volumes"]
    assert any(m.endswith("/app/data/raw:ro") for m in mounts)


# ---------------------------------------------------------------------------
# Berkas ignore — sintaksnya sunyi kalau salah
# ---------------------------------------------------------------------------
def test_tidak_ada_komentar_di_akhir_baris_ignore():
    """Regresi: .gitignore dan .dockerignore TIDAK mengenal komentar inline.

    Baris `reports/*.json   # hasil monitoring` tidak berarti "abaikan
    reports/*.json". Seluruh baris dibaca sebagai satu pola — termasuk spasi
    dan teks komentarnya — sehingga polanya tidak pernah cocok dengan apa pun.
    Tidak ada peringatan, tidak ada error; berkas yang dikira aman diam-diam
    ikut ter-commit. Komentar harus berdiri di barisnya sendiri.
    """
    rusak = []
    for nama in (".gitignore", ".dockerignore"):
        for i, baris in enumerate((ROOT / nama).read_text().splitlines(), start=1):
            if baris.lstrip().startswith("#") or not baris.strip():
                continue
            if re.search(r"\S\s+#", baris):
                rusak.append(f"{nama}:{i}: {baris}")
    assert not rusak, "komentar inline tidak berlaku sebagai komentar:\n" + "\n".join(rusak)


def test_pola_penting_benar_benar_mengabaikan():
    """Uji hasilnya, bukan sintaksnya: apakah berkas rahasia & hasil kerja diabaikan."""
    import subprocess

    if not (ROOT / ".git").exists():
        pytest.skip("belum ada repositori git")

    wajib_diabaikan = [".env", "data/raw/train.csv", "models/house_price_champion.joblib",
                       "mlflow.db", "logs/predictions.log", "reports/drift_report.json",
                       "reports/model_comparison.json"]
    hasil = subprocess.run(["git", "check-ignore", *wajib_diabaikan],
                           cwd=ROOT, capture_output=True, text=True, check=False)
    lolos = set(hasil.stdout.split())
    bocor = [p for p in wajib_diabaikan if p not in lolos]
    assert not bocor, f"TIDAK diabaikan Git: {bocor}"
