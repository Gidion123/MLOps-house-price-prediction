"""Uji modul tracking — bagian yang bisa diuji TANPA menyalakan server MLflow.

Filosofi test di sini: yang diuji adalah keputusan kita, bukan MLflow-nya.
MLflow sudah punya test suite sendiri. Yang bisa salah di project ini adalah
(a) URI relatif yang diam-diam bikin dua database, (b) fungsi log_* yang
meledak ketika tracking dimatikan, dan (c) hash kontrak yang tidak berubah
padahal kontraknya berubah.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import trainer
from src.models.trainer import QualityGateError
from src.utils import config, tracking


# ---------------------------------------------------------------------------
# URI & reproducibility
# ---------------------------------------------------------------------------
def test_uri_relatif_jadi_absolut():
    """sqlite:///mlflow.db harus menempel ke root project, bukan ke cwd."""
    uri = tracking.resolve_tracking_uri("sqlite:///mlflow.db")
    assert uri.startswith("sqlite:////")
    assert str(config.PROJECT_ROOT) in uri


def test_uri_absolut_tidak_diubah():
    asli = "sqlite:////tmp/eksperimen.db"
    assert tracking.resolve_tracking_uri(asli) == asli


def test_uri_non_sqlite_dibiarkan():
    """URI server MLflow sungguhan tidak boleh diutak-atik."""
    asli = "http://localhost:5000"
    assert tracking.resolve_tracking_uri(asli) == asli


def test_file_hash_konsisten_dan_aman():
    h1 = tracking.file_hash(config.TRAIN_CSV)
    h2 = tracking.file_hash(config.TRAIN_CSV)
    assert h1 == h2 and len(h1) == 12
    assert tracking.file_hash(Path("/berkas/yang/tidak/ada.csv")) == "tidak-ada"


def test_git_commit_selalu_mengembalikan_teks():
    """Tidak ada Git bukan alasan training gagal."""
    assert isinstance(tracking.git_commit(), str)


# ---------------------------------------------------------------------------
# Gagal dengan lembut: tanpa run aktif, semua log_* harus diam saja
# ---------------------------------------------------------------------------
def test_start_run_nonaktif_menghasilkan_none():
    with tracking.start_run(nama_run="tidak-dipakai", aktif=False) as run:
        assert run is None


def test_log_tanpa_run_tidak_meledak():
    tracking.log_params({"a": 1})
    tracking.log_metrics({"b": 2.0}, step=1)
    tracking.set_tags({"c": "d"})
    tracking.log_dict({"e": 1}, "f.json")
    tracking.log_artifact(config.CONFIG_FILE, "config")
    assert tracking.log_model(object(), None) is None


def test_alias_champion_satu_sumber():
    assert tracking.ALIAS_CHAMPION == "champion"


# ---------------------------------------------------------------------------
# Hash kontrak & parameter run
# ---------------------------------------------------------------------------
def test_kontrak_hash_berubah_kalau_kontrak_berubah(monkeypatch):
    sebelum = trainer.kontrak_hash()
    monkeypatch.setattr(config, "MODEL_COLUMNS", config.MODEL_COLUMNS + ["KolomBaru"])
    assert trainer.kontrak_hash() != sebelum
    assert len(sebelum) == 12


def test_kontrak_hash_berubah_kalau_daftar_kategori_berubah(monkeypatch):
    """Menambah pilihan kategori juga mengubah soal yang dijawab model."""
    sebelum = trainer.kontrak_hash()
    baru = {k: list(v) for k, v in config.CATEGORICAL_VALUES.items()}
    baru["ExterQual"] = baru["ExterQual"] + ["Po"]
    monkeypatch.setattr(config, "CATEGORICAL_VALUES", baru)
    assert trainer.kontrak_hash() != sebelum


def test_params_run_memuat_semua_hyperparameter():
    params = trainer._params_run("xgboost", True, 1168, 290)
    for nama in config.HYPERPARAMETERS:
        assert f"hp_{nama}" in params
    assert params["n_kolom_model"] == len(config.MODEL_COLUMNS)
    assert params["random_state"] == config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Gerbang mutu tetap galak walau MLflow menyala
# ---------------------------------------------------------------------------
def test_gate_menolak_model_jelek():
    jelek = {"cv_rmse_log": 0.99, "cv_rmse_log_std": 0.01}
    with pytest.raises(QualityGateError):
        trainer.quality_gate(jelek, config.MODEL_COLUMNS)


def test_gate_menolak_kolom_leakage():
    bagus = {"cv_rmse_log": 0.12, "cv_rmse_log_std": 0.01}
    with pytest.raises(QualityGateError):
        trainer.quality_gate(bagus, config.MODEL_COLUMNS + [config.LEAKAGE_COLUMNS[0]])


def test_gate_lolos_untuk_champion_sekarang():
    bagus = {"cv_rmse_log": 0.12383, "cv_rmse_log_std": 0.015}
    checks = trainer.quality_gate(bagus, config.MODEL_COLUMNS)
    assert len(checks) == 5 and all(c["lolos"] for c in checks)


# ---------------------------------------------------------------------------
# Aturan promosi champion — bagian paling berisiko di seluruh sistem
#
# Fungsi ini yang memutuskan model mana yang melayani orang. Karena ia sengaja
# dibuat MURNI (tanpa MLflow, tanpa database, tanpa melatih apa pun), aturannya
# bisa diuji dalam milidetik — dan itu alasan utama ia dipisah.
# ---------------------------------------------------------------------------
def test_champion_pertama_menang_otomatis():
    k = tracking.keputusan_promosi(skor_baru=0.124, std_baru=0.015, skor_lama=None)
    assert k["dipromosikan"] and k["alasan"] == "champion pertama"


def test_menang_jauh_dipromosikan():
    k = tracking.keputusan_promosi(skor_baru=0.110, std_baru=0.015, skor_lama=0.124)
    assert k["dipromosikan"]
    assert k["selisih"] > k["margin"]


def test_menang_tipis_DITOLAK_sebagai_kebisingan():
    """Inti perbaikannya: menang 0.0001 pada model ber-std 0.015 itu keberuntungan.

    Tanpa saringan ini, sistem otomatis akan bolak-balik mengganti champion
    karena ragam sampling — tiap pergantian menanggung risiko produksi tanpa
    dibayar perbaikan apa pun.
    """
    k = tracking.keputusan_promosi(skor_baru=0.12390, std_baru=0.015, skor_lama=0.12400)
    assert not k["dipromosikan"]
    assert "kebisingan" in k["alasan"]


def test_kalah_ditolak():
    k = tracking.keputusan_promosi(skor_baru=0.140, std_baru=0.015, skor_lama=0.124)
    assert not k["dipromosikan"] and "kalah" in k["alasan"]


def test_margin_ikut_membesar_kalau_model_lebih_berisik():
    kecil = tracking.keputusan_promosi(0.120, 0.005, 0.124)["margin"]
    besar = tracking.keputusan_promosi(0.120, 0.050, 0.124)["margin"]
    assert besar > kecil, "model yang lebih berisik harus dituntut menang lebih jauh"


def test_kontrak_berbeda_ditolak_walau_skornya_jauh_lebih_baik():
    """Model 79-fitur vs 20-fitur menjawab soal berbeda — RMSE-nya tidak sebanding."""
    k = tracking.keputusan_promosi(skor_baru=0.050, std_baru=0.015, skor_lama=0.124,
                                   kontrak_baru="aaaaaaaa", kontrak_lama="bbbbbbbb")
    assert not k["dipromosikan"] and "kontrak berbeda" in k["alasan"]


def test_paksa_melewati_semua_saringan():
    """Jalan darurat yang sah: rollback saat produksi bermasalah."""
    k = tracking.keputusan_promosi(skor_baru=0.900, std_baru=0.015, skor_lama=0.124,
                                   kontrak_baru="aaa", kontrak_lama="bbb", paksa=True)
    assert k["dipromosikan"] and "dipaksa" in k["alasan"]


def test_margin_nol_mengembalikan_perilaku_lama(monkeypatch):
    monkeypatch.setattr(config, "PROMOSI_MARGIN_STD_FAKTOR", 0.0)
    k = tracking.keputusan_promosi(skor_baru=0.12399, std_baru=0.015, skor_lama=0.12400)
    assert k["dipromosikan"], "margin 0 berarti menang sekecil apa pun tetap promosi"


def test_pemeriksaan_kontrak_bisa_dimatikan(monkeypatch):
    monkeypatch.setattr(config, "PROMOSI_WAJIB_KONTRAK_SAMA", False)
    k = tracking.keputusan_promosi(skor_baru=0.050, std_baru=0.015, skor_lama=0.124,
                                   kontrak_baru="aaa", kontrak_lama="bbb")
    assert k["dipromosikan"]


def test_alias_dan_fungsi_registry_tersedia():
    for nama in ("tetapkan_champion", "rollback_champion", "status_registry",
                 "muat_bundle_champion", "arahkan_tracking"):
        assert hasattr(tracking, nama), f"{nama} hilang"
