"""metrics.py — metrik Prometheus. Log menjawab "apa yang terjadi pada request itu",
metrik menjawab "bagaimana keadaan sistem sekarang".

Dua hal yang sering dikira sama padahal beda tugas:

* **Log** (``logger.py``)  — satu baris per kejadian, lengkap, mahal disimpan,
  dipakai menelusuri SATU kasus. "request_id abc123 kenapa 500?"
* **Metrik** (file ini)    — angka agregat, murah, dipakai melihat SEMUA kasus
  sekaligus. "sejak deploy tadi, berapa persen request yang 422?"

Grafana menggambar yang kedua. Kamu membaca yang pertama setelah Grafana
menunjukkan ada yang aneh.

Kenapa histogram, bukan rata-rata? Karena rata-rata latency adalah metrik yang
paling sering menipu: 99 request 5 ms + 1 request 3 detik menghasilkan rata-rata
35 ms yang terlihat sehat, padahal ada user yang menunggu 3 detik. Histogram
menyimpan sebarannya, sehingga p95 dan p99 bisa dihitung Prometheus.

Catatan penyebaran: registry ini in-process. Kalau suatu hari uvicorn dijalankan
dengan ``--workers 4``, tiap worker punya angkanya sendiri dan hasil scrape jadi
acak antar-worker. Solusinya ``prometheus_client`` mode multiprocess. Di project
ini sengaja satu worker, jadi belum diperlukan — tapi jangan sampai lupa.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Lalu lintas HTTP
# ---------------------------------------------------------------------------
# Label "path" sengaja memakai TEMPLATE rute (/predict), bukan URL mentah.
# Kalau memakai URL mentah, tiap id unik melahirkan satu deret waktu baru dan
# Prometheus meledak — namanya cardinality explosion, dan ini penyebab nomor
# satu server monitoring tumbang di produksi.
http_requests_total = Counter(
    "house_price_http_requests_total",
    "Jumlah request HTTP yang masuk",
    ["method", "path", "status"],
)

http_request_duration = Histogram(
    "house_price_http_request_duration_seconds",
    "Durasi request HTTP",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
prediksi_total = Counter(
    "house_price_prediksi_total",
    "Jumlah prediksi yang dihasilkan",
    ["hasil"],  # sukses | ditolak | gagal
)

prediksi_durasi = Histogram(
    "house_price_prediksi_duration_seconds",
    "Waktu inference murni (tanpa overhead HTTP)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# Sebaran harga yang dikeluarkan model. Ini metrik drift paling murah yang ada:
# kalau bentuk histogram ini bergeser padahal modelnya sama, yang berubah adalah
# rumah-rumah yang ditanyakan user — dan itu perlu diketahui sebelum akurasinya
# turun diam-diam.
prediksi_harga_usd = Histogram(
    "house_price_prediksi_harga_usd",
    "Sebaran harga hasil prediksi (USD)",
    buckets=(50_000, 100_000, 150_000, 200_000, 250_000,
             300_000, 400_000, 500_000, 750_000, 1_000_000),
)

# ---------------------------------------------------------------------------
# Keadaan model & hasil pemantauan drift
# ---------------------------------------------------------------------------
model_siap = Gauge("house_price_model_siap", "1 kalau artifact model sudah dimuat")

model_metrik = Gauge(
    "house_price_model_metrik",
    "Metrik mutu model yang sedang melayani (dari artifact)",
    ["nama"],
)

model_dilatih_pada_baris = Gauge(
    "house_price_model_training_rows", "Jumlah baris data latih model aktif"
)

# Diisi oleh src/monitor.py, bukan oleh request. Lihat catatan di monitor.py
# soal kenapa drift dihitung terjadwal dan bukan tiap scrape.
drift_fitur_bergeser = Gauge(
    "house_price_drift_fitur_bergeser", "Jumlah fitur yang terdeteksi bergeser"
)

drift_psi = Gauge(
    "house_price_drift_psi", "Population Stability Index per fitur", ["fitur"]
)

drift_sampel = Gauge(
    "house_price_drift_sampel_live", "Jumlah prediksi yang dipakai analisis drift"
)

# Versi model yang sedang aktif di proses ini. Angka, bukan label — supaya
# grafiknya bisa digambar sebagai garis yang "melompat" saat model berganti,
# dan bisa disandingkan dengan grafik error atau latency untuk melihat apakah
# pergantian itu memperbaiki atau memperburuk keadaan.
model_versi_aktif = Gauge(
    "house_price_model_version", "Nomor versi model yang sedang dilayani (0 kalau dari berkas)"
)

# Label "hasil" sengaja hanya berisi 3 nilai tetap: sukses | gagal | tidak_perlu.
# Label dengan nilai bebas (misalnya pesan error) akan meledakkan cardinality —
# tiap pesan unik melahirkan satu deret waktu baru.
model_reload_total = Counter(
    "house_price_model_reload_total",
    "Jumlah percobaan memuat ulang model",
    ["hasil"],
)

model_switch_terakhir = Gauge(
    "house_price_model_switch_terakhir_unix",
    "Kapan model terakhir benar-benar berganti (unix time)",
)

drift_terakhir_dihitung = Gauge(
    "house_price_drift_terakhir_unix", "Kapan analisis drift terakhir dijalankan (unix time)"
)


def catat_model(bundle: dict) -> None:
    """Salin metrik mutu dari artifact ke Prometheus saat model dimuat.

    Gunanya: di Grafana, garis "RMSE model yang sedang jalan" bisa disandingkan
    dengan garis trafik. Saat ada yang deploy model baru, pergeseran garisnya
    terlihat, bukan cuma tertulis di metadata JSON yang tak pernah dibuka.
    """
    model_siap.set(1)
    model_dilatih_pada_baris.set(float(bundle.get("training_rows", 0)))
    for nama in ("cv_rmse_log", "cv_mae_usd", "cv_r2",
                 "holdout_rmse_log", "holdout_mae_usd", "holdout_r2"):
        if nama in bundle:
            model_metrik.labels(nama=nama).set(float(bundle[nama]))


def render() -> tuple[bytes, str]:
    """Isi endpoint /metrics dalam format teks yang dibaca Prometheus."""
    return generate_latest(), CONTENT_TYPE_LATEST
