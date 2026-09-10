"""monitor.py — deteksi drift: apakah data yang masuk hari ini masih mirip data latih?

Jalankan::

    python -m src.monitor                 # laporan ke layar + reports/drift_report.json
    python -m src.monitor --window 500    # pakai 500 prediksi terakhir

Kenapa ini ada. Model tidak rusak; **dunianya** yang berubah. Model ini belajar
dari 1.460 rumah di Ames, Iowa, era 2006-2010. Kalau besok yang bertanya
kebanyakan rumah mewah 4.000 sq ft, tidak ada satu pun error yang muncul: API
tetap 200, latency tetap 6 ms, dan prediksinya tetap keluar — hanya saja makin
sering salah. Itu satu-satunya kegagalan produksi yang tidak bisa ditangkap
unit test maupun health check.

TIGA JENIS DRIFT, dan mana yang bisa kita lihat:

* **Data drift**      — sebaran fitur masuk berubah.        BISA dideteksi di sini.
* **Prediction drift** — sebaran output model berubah.       BISA dideteksi di sini.
* **Concept drift**   — hubungan fitur→harga yang berubah
  (misal pasar crash: rumah yang sama jadi lebih murah).    **TIDAK BISA**, karena
  butuh harga jual sungguhan yang baru diketahui berbulan-bulan kemudian.

Jangan pernah mengklaim monitoring ini "menjamin model tetap akurat". Yang
dijaminnya lebih sempit: kalau INPUT-nya berubah, kita tahu lebih dulu.

DUA ALAT UKUR yang dipakai, sengaja berpasangan:

* **Uji statistik** (Kolmogorov-Smirnov untuk numerik, chi-square untuk
  kategorikal) menjawab "apakah perbedaannya nyata atau kebetulan?" lewat
  p-value. Kelemahannya: dengan sampel besar, perbedaan sekecil apa pun jadi
  "signifikan".
* **PSI** (Population Stability Index) menjawab "seberapa BESAR pergeserannya?"
  Ambang lazim industri: <0.10 stabil, 0.10-0.25 sedang, >0.25 besar. Ini
  aturan jempol, bukan teorema.

Fitur baru dianggap benar-benar bergeser kalau **keduanya** setuju. Satu alat
saja terlalu berisik untuk dijadikan alarm.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from src.data.data_loader import read_csv_honest
from src.data.data_preprocessor import nan_ke_token_kontrak
from src.utils import config
from src.utils.logger import get_logger
from src.utils.metrics import (
    drift_fitur_bergeser,
    drift_psi,
    drift_sampel,
    drift_terakhir_dihitung,
)

logger = get_logger("house_price.monitor")

EPS = 1e-6
PSI_SEDANG = 0.10
PSI_BESAR = 0.25
MIN_SAMPEL = 30  # di bawah ini, uji statistiknya tidak punya daya. Jujur saja.


# ---------------------------------------------------------------------------
# Sumber data
# ---------------------------------------------------------------------------
def baca_prediksi(window: int | None = None, hanya_produksi: bool = True) -> pd.DataFrame:
    """Ambil baris log ber-event ``prediksi`` — inilah data produksi kita.

    Log JSON per baris memang dipilih sejak awal supaya bisa dibaca begini
    tanpa parsing manual (lihat ``src/utils/logger.py``).

    ``hanya_produksi=True`` membuang request yang datang dengan header
    ``X-Traffic-Source`` — trafik test, smoke test, dan health check. Ini bukan
    kerapian: seribu request uji berisi rumah yang sama persis akan terbaca
    sebagai perubahan besar pada dunia nyata. Alarmnya benar secara statistik
    dan salah secara makna, dan alarm semacam itu yang membuat orang berhenti
    membaca laporan drift.

    Baris log lama yang belum punya field ``klien`` dianggap produksi —
    menganggapnya bukan produksi akan menghapus seluruh riwayat begitu kode
    ini di-deploy.
    """
    if not config.PREDICTION_LOG.exists():
        return pd.DataFrame()

    baris = []
    with config.PREDICTION_LOG.open(encoding="utf-8") as f:
        for teks in f:
            teks = teks.strip()
            if not teks.startswith("{") or '"prediksi"' not in teks:
                continue
            try:
                catatan = json.loads(teks)
            except json.JSONDecodeError:
                continue  # baris rusak (mis. proses mati saat menulis) dilewati
            if catatan.get("event") == "prediksi":
                baris.append(catatan)

    df = pd.DataFrame(baris)
    if df.empty:
        return df
    if hanya_produksi:
        asal = df["klien"] if "klien" in df.columns else pd.Series("produksi", index=df.index)
        df = df[asal.fillna("produksi") == "produksi"]
    window = window or config.MONITOR_WINDOW_SIZE
    return df.tail(window).reset_index(drop=True)


def referensi() -> pd.DataFrame:
    """Data latih sebagai pembanding — kolom kontrak saja, plus harga aslinya.

    Penyamaan token kontrak WAJIB ikut diterapkan di sini. Ini bukan detail
    kecil: tanpa itu, ``BsmtQual`` di referensi masih berisi NaN sementara log
    request berisi kata ``"None"`` — dua penamaan untuk hal yang sama.
    Chi-square membacanya sebagai dua kategori yang sama sekali berbeda dan
    melaporkan drift **setiap hari, selamanya**, padahal tidak ada yang
    berubah. Alarm yang selalu berbunyi sama saja dengan tidak ada alarm.

    Aturan umumnya: referensi harus melihat data dengan mata yang sama seperti
    jalur serving. Sama seperti training-serving skew, versi monitoring.
    """
    train = read_csv_honest(config.TRAIN_CSV)
    ref = nan_ke_token_kontrak(train)[config.API_CONTRACT].copy()
    ref["prediksi"] = train[config.TARGET].astype(float)
    return ref


# ---------------------------------------------------------------------------
# Alat ukur
# ---------------------------------------------------------------------------
def psi_numerik(ref: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """PSI untuk kolom numerik. Bin dibuat dari kuantil DATA REFERENSI.

    Bin harus berasal dari referensi, bukan dari gabungan keduanya — kalau
    dihitung dari gabungan, pergeseran ikut menggeser binnya sendiri dan
    PSI-nya mengecil justru saat drift-nya besar.
    """
    tepi = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(tepi) < 3:
        return 0.0
    tepi[0], tepi[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, bins=tepi)[0] / max(len(ref), 1) + EPS
    p_live = np.histogram(live, bins=tepi)[0] / max(len(live), 1) + EPS
    return float(np.sum((p_live - p_ref) * np.log(p_live / p_ref)))


def psi_kategorikal(ref: pd.Series, live: pd.Series) -> float:
    kategori = sorted(set(ref.astype(str)) | set(live.astype(str)))
    p_ref = ref.astype(str).value_counts(normalize=True).reindex(kategori).fillna(0) + EPS
    p_live = live.astype(str).value_counts(normalize=True).reindex(kategori).fillna(0) + EPS
    return float(np.sum((p_live - p_ref) * np.log(p_live / p_ref)))


def _label(psi: float) -> str:
    if psi >= PSI_BESAR:
        return "besar"
    return "sedang" if psi >= PSI_SEDANG else "stabil"


def uji_numerik(nama: str, ref: pd.Series, live: pd.Series) -> dict[str, Any]:
    r, l = ref.dropna().to_numpy(float), live.dropna().to_numpy(float)
    stat, p = ks_2samp(r, l)
    nilai_psi = psi_numerik(r, l)
    return {"fitur": nama, "tipe": "numerik", "uji": "kolmogorov-smirnov",
            "statistik": round(float(stat), 4), "p_value": float(p),
            "psi": round(nilai_psi, 4), "besaran": _label(nilai_psi),
            "rata_ref": round(float(np.mean(r)), 2), "rata_live": round(float(np.mean(l)), 2)}


def uji_kategorikal(nama: str, ref: pd.Series, live: pd.Series) -> dict[str, Any]:
    kategori = sorted(set(ref.astype(str)) | set(live.astype(str)))
    n_live = len(live)
    diamati = live.astype(str).value_counts().reindex(kategori).fillna(0).to_numpy(float)
    harapan = (ref.astype(str).value_counts(normalize=True)
               .reindex(kategori).fillna(0).to_numpy(float)) * n_live
    # chi-square menuntut total keduanya sama persis; epsilon dipakai supaya
    # kategori yang tidak pernah muncul di referensi tidak membuat pembagian nol.
    harapan = np.maximum(harapan, EPS)
    harapan = harapan * (diamati.sum() / harapan.sum())
    stat, p = chisquare(f_obs=diamati, f_exp=harapan)
    nilai_psi = psi_kategorikal(ref, live)
    baru = sorted(set(live.astype(str)) - set(ref.astype(str)))
    return {"fitur": nama, "tipe": "kategorikal", "uji": "chi-square",
            "statistik": round(float(stat), 4), "p_value": float(p),
            "psi": round(nilai_psi, 4), "besaran": _label(nilai_psi),
            "kategori_baru": baru}


# ---------------------------------------------------------------------------
# Analisis
# ---------------------------------------------------------------------------
def analisis_drift(window: int | None = None, bonferroni: bool = True,
                   hanya_produksi: bool = True) -> dict[str, Any]:
    """Bandingkan prediksi terakhir dengan data latih, fitur per fitur.

    ``bonferroni=True`` membagi ambang p dengan jumlah fitur yang diuji.
    Alasannya penting: menguji 21 fitur pada alpha 0.05 berarti kita
    MENGHARAPKAN sekitar satu alarm palsu tiap kali laporan dijalankan. Tim
    yang tiga hari berturut-turut mengejar alarm palsu akan berhenti membaca
    laporannya sama sekali — dan itu lebih buruk daripada tidak punya monitoring.
    """
    live = baca_prediksi(window, hanya_produksi=hanya_produksi)
    hasil: dict[str, Any] = {
        "dihitung_pada": pd.Timestamp.now().isoformat(timespec="seconds"),
        "window_diminta": window or config.MONITOR_WINDOW_SIZE,
        "sampel_live": int(len(live)),
        "hanya_trafik_produksi": hanya_produksi,
    }

    if len(live) < MIN_SAMPEL:
        hasil.update({"status": "data_kurang",
                      "catatan": f"butuh minimal {MIN_SAMPEL} prediksi, baru ada {len(live)}",
                      "fitur": []})
        logger.warning("drift_dilewati", extra={"sampel": len(live), "minimal": MIN_SAMPEL})
        drift_sampel.set(len(live))
        return hasil

    ref = referensi()
    kolom = [k for k in config.API_CONTRACT if k in live.columns]
    if "prediksi" in live.columns:
        kolom.append("prediksi")  # prediction drift ikut diuji

    alpha = config.DRIFT_P_VALUE_THRESHOLD / (len(kolom) if bonferroni else 1)
    laporan = []
    for nama in kolom:
        if nama in config.API_CONTRACT_CATEGORICAL:
            baris = uji_kategorikal(nama, ref[nama], live[nama])
        else:
            baris = uji_numerik(nama, ref[nama], live[nama])
        # Alarm hanya kalau uji statistik DAN besaran PSI sama-sama setuju.
        baris["signifikan"] = bool(baris["p_value"] < alpha)
        baris["bergeser"] = bool(baris["signifikan"] and baris["psi"] >= PSI_SEDANG)
        baris["p_value"] = round(baris["p_value"], 6)
        laporan.append(baris)

    bergeser = [b["fitur"] for b in laporan if b["bergeser"]]
    hasil.update({
        "status": "drift_terdeteksi" if bergeser else "stabil",
        "alpha_efektif": round(alpha, 6),
        "koreksi_bonferroni": bonferroni,
        "fitur_diuji": len(kolom),
        "fitur_bergeser": bergeser,
        "fitur": sorted(laporan, key=lambda b: -b["psi"]),
    })

    # Metrik Prometheus diperbarui di sini — lihat catatan di bawah soal kenapa
    # drift dihitung terjadwal dan bukan tiap kali /metrics di-scrape.
    drift_fitur_bergeser.set(len(bergeser))
    drift_sampel.set(len(live))
    drift_terakhir_dihitung.set(time.time())
    for baris in laporan:
        drift_psi.labels(fitur=baris["fitur"]).set(baris["psi"])

    logger.info("drift_dianalisis", extra={"status": hasil["status"],
                                           "sampel": len(live),
                                           "bergeser": bergeser})
    return hasil


def simpan_laporan(hasil: dict[str, Any]) -> None:
    config.ensure_directories()
    tujuan = config.PROJECT_ROOT / "reports" / "drift_report.json"
    tujuan.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("laporan_drift_disimpan", extra={"path": str(tujuan)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis drift dari log prediksi")
    parser.add_argument("--window", type=int, default=None,
                        help=f"jumlah prediksi terakhir (default {config.MONITOR_WINDOW_SIZE})")
    parser.add_argument("--tanpa-bonferroni", action="store_true",
                        help="matikan koreksi banyak-uji (lebih sensitif, lebih berisik)")
    parser.add_argument("--semua-klien", action="store_true",
                        help="ikutkan trafik test/health check (default: hanya produksi)")
    args = parser.parse_args()

    hasil = analisis_drift(args.window, bonferroni=not args.tanpa_bonferroni,
                           hanya_produksi=not args.semua_klien)
    simpan_laporan(hasil)

    print("\n" + "=" * 74)
    print(f"LAPORAN DRIFT — {hasil['dihitung_pada']}")
    print("=" * 74)
    print(f"  sampel produksi : {hasil['sampel_live']}")
    if hasil["status"] == "data_kurang":
        print(f"  status          : DATA KURANG — {hasil['catatan']}")
        return

    print(f"  fitur diuji     : {hasil['fitur_diuji']}  "
          f"(alpha efektif {hasil['alpha_efektif']}"
          f"{', Bonferroni' if hasil['koreksi_bonferroni'] else ''})")
    print(f"  status          : {hasil['status'].upper()}")
    print("-" * 74)
    print(f"  {'fitur':<16}{'uji':<8}{'p-value':>10}{'PSI':>9}  {'besaran':<9}bergeser")
    print("-" * 74)
    for b in hasil["fitur"]:
        print(f"  {b['fitur']:<16}{b['uji'][:3]:<8}{b['p_value']:>10.5f}{b['psi']:>9.4f}  "
              f"{b['besaran']:<9}{'YA' if b['bergeser'] else '-'}")
    print("-" * 74)
    if hasil["fitur_bergeser"]:
        print(f"  PERLU DITINDAK: {hasil['fitur_bergeser']}")
        print("  Langkah lazim: periksa sumber trafik dulu (apakah ada klien baru?),")
        print("  baru pertimbangkan melatih ulang dengan data terbaru.")
    else:
        print("  Tidak ada fitur yang bergeser signifikan DAN besar.")


if __name__ == "__main__":
    main()
