"""Uji detektor drift.

Satu hal yang harus dipegang saat menguji monitoring: **detektor yang tidak
pernah berbunyi kelihatan sama persis dengan sistem yang sehat.** Karena itu
tiap uji di sini berpasangan — satu memastikan ia DIAM saat data normal, satu
memastikan ia BERBUNYI saat data digeser. Yang pertama saja tidak membuktikan
apa pun.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import monitor
from src.utils import config


@pytest.fixture(scope="module")
def ref():
    return monitor.referensi()


# ---------------------------------------------------------------------------
# Referensi harus melihat data seperti jalur serving melihatnya
# ---------------------------------------------------------------------------
def test_referensi_memakai_token_kontrak_yang_sama(ref):
    """Regresi: BsmtQual di referensi tidak boleh NaN.

    Kalau NaN, ia tidak akan pernah cocok dengan "None" yang dikirim API,
    dan detektor melaporkan drift palsu setiap hari selamanya.
    """
    assert ref["BsmtQual"].isna().sum() == 0
    assert config.STRUCTURAL_API_TOKEN in set(ref["BsmtQual"])


def test_referensi_berisi_kolom_kontrak_dan_target(ref):
    assert set(config.API_CONTRACT).issubset(ref.columns)
    assert "prediksi" in ref.columns


# ---------------------------------------------------------------------------
# PSI: diam saat sama, berbunyi saat digeser
# ---------------------------------------------------------------------------
def test_psi_numerik_nol_untuk_sebaran_sama():
    x = np.random.default_rng(0).normal(size=2000)
    assert monitor.psi_numerik(x, x) < 0.01


def test_psi_numerik_besar_untuk_sebaran_bergeser():
    rng = np.random.default_rng(0)
    assert monitor.psi_numerik(rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)) > monitor.PSI_BESAR


def test_psi_kategorikal_membedakan_komposisi():
    a = pd.Series(["x"] * 80 + ["y"] * 20)
    assert monitor.psi_kategorikal(a, a) < 0.01
    assert monitor.psi_kategorikal(a, pd.Series(["x"] * 20 + ["y"] * 80)) > monitor.PSI_BESAR


def test_label_besaran_mengikuti_ambang():
    assert monitor._label(0.05) == "stabil"
    assert monitor._label(0.15) == "sedang"
    assert monitor._label(0.40) == "besar"


# ---------------------------------------------------------------------------
# Uji per fitur
# ---------------------------------------------------------------------------
def test_uji_numerik_tidak_menuduh_sebaran_sama(ref):
    hasil = monitor.uji_numerik("GrLivArea", ref["GrLivArea"], ref["GrLivArea"])
    assert hasil["p_value"] > 0.05 and hasil["psi"] < monitor.PSI_SEDANG


def test_uji_kategorikal_melaporkan_kategori_baru(ref):
    live = pd.Series(["Medan"] * 50)
    hasil = monitor.uji_kategorikal("Neighborhood", ref["Neighborhood"], live)
    assert hasil["kategori_baru"] == ["Medan"]
    assert hasil["psi"] > monitor.PSI_BESAR


# ---------------------------------------------------------------------------
# Analisis penuh
# ---------------------------------------------------------------------------
def test_data_kurang_tidak_dipaksakan(monkeypatch):
    """Sampel sedikit harus dilaporkan apa adanya, bukan dijawab 'stabil'."""
    monkeypatch.setattr(monitor, "baca_prediksi", lambda *a, **k: pd.DataFrame([{"a": 1}] * 5))
    hasil = monitor.analisis_drift()
    assert hasil["status"] == "data_kurang" and hasil["fitur"] == []


def test_trafik_seperti_data_latih_dinyatakan_stabil(ref, monkeypatch):
    live = ref.sample(200, random_state=0).reset_index(drop=True)
    monkeypatch.setattr(monitor, "baca_prediksi", lambda *a, **k: live)
    hasil = monitor.analisis_drift()
    assert hasil["status"] == "stabil" and hasil["fitur_bergeser"] == []


def test_trafik_rumah_mewah_terdeteksi_bergeser(ref, monkeypatch):
    mewah = ref[(ref.OverallQual >= 8) & (ref.GrLivArea > 2000)]
    live = mewah.sample(200, replace=True, random_state=0).reset_index(drop=True)
    monkeypatch.setattr(monitor, "baca_prediksi", lambda *a, **k: live)
    hasil = monitor.analisis_drift()
    assert hasil["status"] == "drift_terdeteksi"
    for wajib in ("OverallQual", "GrLivArea", "prediksi"):
        assert wajib in hasil["fitur_bergeser"]


def test_bonferroni_memperketat_ambang(ref, monkeypatch):
    live = ref.sample(200, random_state=1).reset_index(drop=True)
    monkeypatch.setattr(monitor, "baca_prediksi", lambda *a, **k: live)
    ketat = monitor.analisis_drift(bonferroni=True)["alpha_efektif"]
    longgar = monitor.analisis_drift(bonferroni=False)["alpha_efektif"]
    assert ketat < longgar == config.DRIFT_P_VALUE_THRESHOLD


def test_hasil_terurut_dari_psi_terbesar(ref, monkeypatch):
    mewah = ref[ref.OverallQual >= 8]
    monkeypatch.setattr(monitor, "baca_prediksi",
                        lambda *a, **k: mewah.sample(200, replace=True, random_state=2))
    fitur = monitor.analisis_drift()["fitur"]
    assert [f["psi"] for f in fitur] == sorted([f["psi"] for f in fitur], reverse=True)


# ---------------------------------------------------------------------------
# Trafik otomatis tidak boleh mencemari analisis drift
# ---------------------------------------------------------------------------
def test_trafik_bertanda_klien_dibuang(tmp_path, monkeypatch):
    """Request ber-header X-Client tidak ikut dihitung sebagai data produksi."""
    berkas = tmp_path / "predictions.log"
    baris = ([json.dumps({"event": "prediksi", "klien": "produksi", "GrLivArea": 1500})] * 4
             + [json.dumps({"event": "prediksi", "klien": "test", "GrLivArea": 9999})] * 6)
    berkas.write_text("\n".join(baris), encoding="utf-8")
    monkeypatch.setattr(config, "PREDICTION_LOG", berkas)

    assert len(monitor.baca_prediksi()) == 4
    assert len(monitor.baca_prediksi(hanya_produksi=False)) == 10


def test_log_lama_tanpa_field_klien_tetap_dihitung(tmp_path, monkeypatch):
    """Baris log yang ditulis sebelum fitur ini ada dianggap produksi.

    Kalau tidak, seluruh riwayat drift hilang begitu kode baru di-deploy.
    """
    berkas = tmp_path / "predictions.log"
    berkas.write_text("\n".join([json.dumps({"event": "prediksi", "GrLivArea": 1500})] * 3),
                      encoding="utf-8")
    monkeypatch.setattr(config, "PREDICTION_LOG", berkas)
    assert len(monitor.baca_prediksi()) == 3
