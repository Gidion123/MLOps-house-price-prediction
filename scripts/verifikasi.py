"""verifikasi.py — satu perintah yang membuktikan seluruh rantai masih nyambung.

    python -m scripts.verifikasi            # cepat (~20 detik)
    python -m scripts.verifikasi --lengkap  # ikut menjalankan CV + quality gate

Kenapa ini ada, padahal sudah ada 111 unit test?

Unit test memeriksa tiap potongan sendiri-sendiri. Yang tidak diperiksanya adalah
SAMBUNGAN antar potongan setelah semuanya di-deploy: apakah kontrak yang dipakai
model, yang divalidasi Pydantic, dan yang diterbitkan OpenAPI masih benda yang
sama? Kegagalan di sambungan tidak memunculkan error — ia memunculkan angka yang
salah dengan status 200.

Skrip ini juga yang dipakai saat demo: satu perintah, satu tabel, semua hijau.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HASIL: list[tuple[str, bool, str]] = []


def periksa(nama: str):
    """Dekorator kecil: jalankan pemeriksaan, tangkap kegagalannya, catat hasilnya."""
    def bungkus(fn):
        def jalan(*a, **k):
            mulai = time.perf_counter()
            try:
                bukti = fn(*a, **k) or "ok"
                lolos = True
            except Exception as err:  # noqa: BLE001 - satu gagal tidak menghentikan sisanya
                bukti, lolos = f"{type(err).__name__}: {err}", False
            HASIL.append((nama, lolos, f"{bukti}  [{(time.perf_counter() - mulai) * 1000:.0f} ms]"))
            return lolos
        return jalan
    return bungkus


# ---------------------------------------------------------------------------
@periksa("Konfigurasi & artifact tersedia")
def cek_artifact():
    from src.utils import config
    assert config.CONFIG_FILE.exists(), "config.yaml hilang"
    assert config.TRAIN_CSV.exists(), "data/raw/train.csv hilang"
    assert config.MODEL_ARTIFACT.exists(), "artifact belum ada — jalankan trainer"
    ukuran = config.MODEL_ARTIFACT.stat().st_size / 1024 / 1024
    return f"artifact {ukuran:.2f} MB, {len(config.API_CONTRACT)} field kontrak"


@periksa("Kontrak identik di 4 tempat")
def cek_kontrak_selaras():
    """config.yaml == artifact == schema Pydantic == OpenAPI.

    Ini pemeriksaan terpenting di seluruh skrip. Kontrak hidup di empat tempat;
    kalau salah satunya melenceng, model menerima kolom yang berbeda dari yang
    dikirim user dan hasilnya tetap berupa angka yang tampak wajar.
    """
    from fastapi.testclient import TestClient

    from api.main import app
    from api.schemas import HousePriceRequest
    from src.models.predict import load_bundle
    from src.utils import config

    dari_config = set(config.API_CONTRACT)
    dari_artifact = set(load_bundle()["api_contract_fields"])
    dari_schema = set(HousePriceRequest.model_fields)
    with TestClient(app) as c:
        spec = c.get("/openapi.json").json()
    dari_openapi = set(spec["components"]["schemas"]["HousePriceRequest"]["properties"])

    assert dari_config == dari_artifact == dari_schema == dari_openapi, (
        f"kontrak melenceng: config={len(dari_config)} artifact={len(dari_artifact)} "
        f"schema={len(dari_schema)} openapi={len(dari_openapi)}"
    )
    return f"{len(dari_config)} field cocok di config, artifact, Pydantic, OpenAPI"


@periksa("Jalur inference (tanpa HTTP)")
def cek_inference():
    from src.models.predict import CONTOH_REQUEST, predict_one
    hasil = predict_one(CONTOH_REQUEST, request_id="verifikasi", klien="verifikasi")
    assert 50_000 < hasil["predicted_price"] < 800_000, "harga di luar nalar"
    return f"${hasil['predicted_price']:,.0f} dalam {hasil['latency_ms']:.1f} ms"


@periksa("API menjawab benar dan menolak yang salah")
def cek_api():
    from fastapi.testclient import TestClient

    from api.main import app
    from src.models.predict import CONTOH_REQUEST

    with TestClient(app) as c:
        # Header X-Traffic-Source menandai trafik ini sebagai bukan-produksi, supaya
        # verifikasi tidak mencemari analisis drift.
        c.headers["X-Traffic-Source"] = "verifikasi"
        assert c.get("/health").json()["model_loaded"] is True
        assert c.get("/ready").status_code == 200
        info = c.get("/model-info").json()

        ok = c.post("/predict", json=CONTOH_REQUEST)
        assert ok.status_code == 200, "request sah ditolak"
        harga = ok.json()["predicted_price_usd"]

        batch = c.post("/predict/batch", json={"houses": [CONTOH_REQUEST] * 3})
        assert batch.json()["count"] == 3

        # Yang menentukan bukan jawaban benarnya, tapi penolakannya.
        for rusak in ({**CONTOH_REQUEST, "OverallQual": 99},
                      {**CONTOH_REQUEST, "Neighborhood": "Medan"},
                      {k: v for k, v in CONTOH_REQUEST.items() if k != "GrLivArea"},
                      {**CONTOH_REQUEST, "GrLivArae": 1}):
            assert c.post("/predict", json=rusak).status_code == 422, "request rusak diterima!"

    return (f"${harga:,.0f} · batch 3 baris · 4 request rusak ditolak 422 · "
            f"model dilatih pada {info['training_rows']:,} baris")


@periksa("Metrik Prometheus terekspos")
def cek_metrik():
    from fastapi.testclient import TestClient

    from api.main import app
    from src.models.predict import CONTOH_REQUEST

    with TestClient(app) as c:
        c.headers["X-Traffic-Source"] = "verifikasi"
        c.post("/predict", json=CONTOH_REQUEST)
        isi = c.get("/metrics").text

    wajib = ["house_price_prediksi_total", "house_price_http_requests_total",
             "house_price_model_metrik", "process_resident_memory_bytes"]
    hilang = [m for m in wajib if m not in isi]
    assert not hilang, f"metrik hilang: {hilang}"
    return f"{len([l for l in isi.splitlines() if l.startswith('# HELP')])} metrik terdaftar"


@periksa("Log terstruktur bisa dibaca sebagai data")
def cek_log():
    import pandas as pd

    from src.utils import config
    assert config.PREDICTION_LOG.exists(), "belum ada log"
    baris = [json.loads(b) for b in config.PREDICTION_LOG.read_text().splitlines()
             if b.strip().startswith("{")]
    df = pd.DataFrame(baris)
    kejadian = df["event"].value_counts().to_dict()
    penting = [k for k in ("prediksi", "http_request", "preprocessing") if k in kejadian]
    assert penting, "tidak ada satu pun kejadian penting tercatat"
    return f"{len(df):,} baris · kejadian: {', '.join(penting)}"


@periksa("Analisis drift berjalan")
def cek_drift():
    from src import monitor
    hasil = monitor.analisis_drift()
    assert hasil["status"] in {"stabil", "drift_terdeteksi", "data_kurang"}
    if hasil["status"] == "data_kurang":
        return f"status: data_kurang ({hasil['sampel_live']} sampel) — kirim request dulu"
    return (f"status: {hasil['status']} · {hasil['sampel_live']} sampel · "
            f"{len(hasil['fitur_bergeser'])} fitur bergeser dari {hasil['fitur_diuji']}")


@periksa("MLflow registry punya champion")
def cek_registry():
    """Baca alias @champion dari registry — TANPA menulis apa pun.

    Sengaja tidak memakai ``tracking.setup()``: fungsi itu memanggil
    ``set_experiment`` yang menulis ke database. Pemeriksaan yang menulis
    adalah pemeriksaan yang bisa merusak benda yang sedang diperiksanya, dan
    pada store SQLite di filesystem yang tidak mengizinkan hapus, tulisan yang
    gagal meninggalkan berkas ``-journal`` yang membuat database TIDAK BISA
    DIBUKA sama sekali sesudahnya.
    """
    from src.utils import config, tracking
    if not tracking.MLFLOW_TERSEDIA:
        raise AssertionError("mlflow tidak terpasang")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking.resolve_tracking_uri())
    nama = config.get_settings().mlflow_model_name
    versi = MlflowClient().get_model_version_by_alias(nama, tracking.ALIAS_CHAMPION)
    return f"{nama} v{versi.version} @{tracking.ALIAS_CHAMPION} (CV {versi.tags.get('cv_rmse_log', '-')})"


@periksa("Pergantian model otomatis siap")
def cek_switching():
    """Periksa mekanismenya ada dan konsisten — tanpa benar-benar mengganti model."""
    import api.service as service
    from src.utils import config

    st = service.status_model()
    assert st["model_dimuat"], "model belum dimuat"
    assert st["identitas_aktif"], "identitas model aktif kosong"
    assert st["perlu_ganti"] is False, (
        f"model aktif ({st['identitas_aktif']}) tertinggal dari champion "
        f"({st['identitas_champion']}) — panggil /admin/reload")
    # Kandidat yang sedang aktif wajib lolos validasinya sendiri.
    service.validasi_bundle(service.muat_model())
    interval = config.get_settings().model_check_interval_seconds
    return (f"sumber={st['sumber']} · aktif={st['identitas_aktif']} · "
            f"pemantau={'nyala ' + str(interval) + 's' if interval > 0 else 'mati'}")


@periksa("Frontend tidak mengimpor backend")
def cek_batas_frontend():
    from src.utils import config
    pola = re.compile(r"^\s*(?:from|import)\s+(?:src|api)\b", re.MULTILINE)
    berkas = list((config.PROJECT_ROOT / "frontend").rglob("*.py"))
    assert berkas, "folder frontend kosong"
    pelanggar = [f.name for f in berkas if pola.search(f.read_text(encoding="utf-8"))]
    assert not pelanggar, f"frontend mengimpor backend: {pelanggar}"
    return f"{len(berkas)} berkas frontend bersih dari import backend"


@periksa("Frontend bisa membangun form dari kontrak API")
def cek_frontend_kontrak():
    from fastapi.testclient import TestClient

    from api.main import app
    from frontend import kontrak
    from src.utils import config

    with TestClient(app) as c:
        spec = c.get("/openapi.json").json()
    bidang = kontrak.bidang_dari_skema(spec)
    assert {b["nama"] for b in bidang} == set(config.API_CONTRACT)
    berpilihan = [b for b in bidang if b["pilihan"]]
    return f"{len(bidang)} field terbaca, {len(berpilihan)} punya daftar pilihan"


@periksa("Jalur training + quality gate (lengkap)")
def cek_training():
    from src.models.trainer import run
    m = run(skip_export=True, track=False)
    return (f"CV {m['cv_rmse_log']:.5f} ± {m['cv_rmse_log_std']:.5f} · "
            f"holdout {m['holdout_rmse_log']:.5f} · R² {m['holdout_r2']:.5f} · gate LOLOS")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Verifikasi end-to-end")
    p.add_argument("--lengkap", action="store_true",
                   help="ikut menjalankan CV + quality gate (menambah ~30 detik)")
    args = p.parse_args()

    cek_artifact()
    cek_kontrak_selaras()
    cek_inference()
    cek_api()
    cek_metrik()
    cek_log()
    cek_drift()
    cek_registry()
    cek_switching()
    cek_batas_frontend()
    cek_frontend_kontrak()
    if args.lengkap:
        cek_training()

    lebar = max(len(n) for n, _, _ in HASIL)
    print("\n" + "=" * 100)
    print("VERIFIKASI END-TO-END — house-price-prediction")
    print("=" * 100)
    for nama, lolos, bukti in HASIL:
        print(f"  {'LOLOS ' if lolos else 'GAGAL '} {nama:<{lebar}}  {bukti}")
    print("=" * 100)

    gagal = [n for n, lolos, _ in HASIL if not lolos]
    if gagal:
        print(f"  {len(gagal)} pemeriksaan GAGAL: {gagal}\n")
        sys.exit(1)
    print(f"  Semua {len(HASIL)} pemeriksaan lolos.\n")


if __name__ == "__main__":
    main()
