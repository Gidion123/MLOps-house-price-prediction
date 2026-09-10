"""tracking.py — jembatan tipis ke MLflow: catat, bandingkan, promosikan.

Kenapa perlu MLflow, padahal metrik sudah dicetak ke layar dan sudah disimpan
di ``models/house_price_champion_metadata.json``?

Karena eksperimen itu jamak. Di project ini saja sudah ada belasan kombinasi:
79 fitur vs 20 fitur, dengan/tanpa fitur turunan, XGB tunggal vs blend, imputasi
modus vs structural. Selama ini angka-angka itu hidup di output sel notebook —
hilang begitu kernel di-restart, dan tidak bisa menjawab pertanyaan paling wajar
di dunia MLOps: *"run yang RMSE-nya 0.12383 itu pakai parameter apa, data versi
mana, dan commit yang mana?"*

MLflow menjawabnya lewat tiga lapis:

* **Tracking**  — tiap run tersimpan utuh: parameter, metrik, artifact, commit
  Git, hash file data. Reproducibility quartet (slide 27) tercatat otomatis.
* **Registry**  — hanya model yang LOLOS gerbang mutu yang didaftarkan, dan
  tiap pendaftaran dapat nomor versi.
* **Alias**     — satu nama tetap, ``@champion``, menunjuk versi yang sedang
  melayani produksi. API memuat lewat alias, bukan nomor versi. Konsekuensinya
  enak: rollback = memindahkan alias, bukan deploy ulang.

Modul ini SENGAJA tipis dan gagal-dengan-lembut. Kalau MLflow tidak terpasang
atau tracking-nya dimatikan, training tetap jalan dan artifact tetap diekspor —
hanya tanpa catatan, dengan peringatan di log. Pipeline training yang mati
gara-gara server pencatat mati itu salah prioritas: yang dicatat kalah penting
dari yang dicatat-tentangnya.
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.utils import config
from src.utils.logger import get_logger

logger = get_logger("house_price.tracking")

# Impor MLflow dijaga: modul lain tetap bisa diimpor walau mlflow belum dipasang
# (misalnya di container API yang tidak butuh tracking sama sekali).
try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    MLFLOW_TERSEDIA = True
except ImportError:  # pragma: no cover - hanya terjadi di environment minimal
    mlflow = None
    MlflowException = Exception
    infer_signature = None
    MlflowClient = None
    MLFLOW_TERSEDIA = False

# Nama alias produksi. Dipakai training (menaruh) dan API (mengambil), jadi
# harus satu konstanta — bukan string "champion" yang diketik ulang di dua file.
ALIAS_CHAMPION = "champion"


# ---------------------------------------------------------------------------
# Bagian 1 — utilitas reproducibility
# ---------------------------------------------------------------------------
def resolve_tracking_uri(uri: str | None = None) -> str:
    """Ubah ``sqlite:///mlflow.db`` yang relatif jadi path absolut.

    Kenapa penting: URI relatif diselesaikan terhadap *current working
    directory*. Jalankan training dari root project dan dari ``notebooks/``,
    hasilnya dua database berbeda yang sama-sama bernama ``mlflow.db`` — dan
    setengah eksperimen kita hilang tanpa satu pun pesan error.

    Catatan format: ``sqlite:///`` + path absolut (yang diawali ``/``) otomatis
    menghasilkan ``sqlite:////...`` berslash empat, dan itu memang ejaan yang
    benar untuk path absolut di SQLAlchemy.
    """
    uri = uri or config.get_settings().mlflow_tracking_uri
    awalan = "sqlite:///"
    if uri.startswith(awalan):
        path = uri[len(awalan):]
        if path and not Path(path).is_absolute():
            uri = awalan + str((config.PROJECT_ROOT / path).resolve())
    return uri


def git_commit() -> str:
    """SHA commit yang sedang aktif — potongan "kode" dari reproducibility quartet."""
    try:
        hasil = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        return hasil.stdout.strip() or "tanpa-git"
    except Exception:  # noqa: BLE001 - git tidak ada bukan alasan training gagal
        return "tanpa-git"


def file_hash(path: Path, panjang: int = 12) -> str:
    """SHA-256 sebuah file — potongan "data" dari reproducibility quartet.

    Kalau suatu hari angka berubah tanpa sebab, hash inilah yang membedakan
    "kodenya berubah" dari "datanya diam-diam berubah".
    """
    if not Path(path).exists():
        return "tidak-ada"
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            digest.update(blok)
    return digest.hexdigest()[:panjang]


# ---------------------------------------------------------------------------
# Bagian 2 — siklus hidup run
# ---------------------------------------------------------------------------
def setup(aktif: bool = True) -> bool:
    """Arahkan MLflow ke database & experiment yang benar. True kalau siap."""
    if not aktif:
        return False
    if not MLFLOW_TERSEDIA:
        logger.warning(
            "mlflow_tidak_terpasang",
            extra={"saran": "pip install mlflow==3.15.1", "dampak": "training jalan tanpa catatan"},
        )
        return False

    pengaturan = config.get_settings()
    uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(pengaturan.mlflow_experiment_name)
    logger.info("mlflow_siap", extra={"tracking_uri": uri,
                                      "experiment": pengaturan.mlflow_experiment_name})
    return True


def _ada_run() -> bool:
    """True kalau saat ini benar-benar ada run MLflow yang terbuka."""
    return MLFLOW_TERSEDIA and mlflow.active_run() is not None


@contextlib.contextmanager
def start_run(nama_run: str | None = None, tags: dict[str, Any] | None = None,
              aktif: bool = True) -> Iterator[Any]:
    """Buka satu run MLflow. Kalau tracking mati, ``yield None`` dan lanjut.

    Semua fungsi ``log_*`` di bawah memeriksa sendiri apakah ada run aktif,
    jadi pemanggil tidak perlu menulis ``if tracking_aktif:`` di mana-mana —
    kode training tetap lurus dibaca.
    """
    if not setup(aktif):
        yield None
        return

    with mlflow.start_run(run_name=nama_run) as run:
        bawaan = {
            "git_commit": git_commit(),
            "data_hash_train": file_hash(config.TRAIN_CSV),
            "proyek": "house-price-prediction",
        }
        mlflow.set_tags({**bawaan, **(tags or {})})
        logger.info("mlflow_run_mulai",
                    extra={"run_id": run.info.run_id, "run_name": nama_run})
        yield run
        logger.info("mlflow_run_selesai", extra={"run_id": run.info.run_id})


def log_params(params: dict[str, Any]) -> None:
    """Catat parameter. Nilai ``None`` dibuang supaya tidak jadi string "None"."""
    if not _ada_run():
        return
    mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Catat metrik numerik. Nilai non-numerik diabaikan diam-diam."""
    if not _ada_run():
        return
    angka = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if angka:
        mlflow.log_metrics(angka, step=step)


def set_tags(tags: dict[str, Any]) -> None:
    if not _ada_run():
        return
    mlflow.set_tags(tags)


def log_dict(obj: Any, nama_file: str) -> None:
    """Simpan dict/list sebagai artifact JSON di dalam run."""
    if not _ada_run():
        return
    mlflow.log_dict(obj, nama_file)


def log_artifact(path: Path, subfolder: str | None = None) -> None:
    if not _ada_run() or not Path(path).exists():
        return
    mlflow.log_artifact(str(path), artifact_path=subfolder)


# ---------------------------------------------------------------------------
# Bagian 3 — model & registry
# ---------------------------------------------------------------------------
def log_model(serving_model: Any, contoh_input: Any, nama_artifact: str = "model") -> str | None:
    """Simpan model end-to-end ke dalam run, lengkap dengan signature.

    Yang disimpan di sini BUKAN model XGBoost telanjang, melainkan
    ``TransformedTargetRegressor(Pipeline(preprocess -> model))``: masuk
    DataFrame 20 field mentah, keluar angka dalam **USD**. Bukan log-USD.

    Ini disengaja. Model registry yang mengeluarkan angka log adalah ranjau:
    siapa pun yang memuatnya lewat ``models:/...@champion`` dan lupa memanggil
    ``expm1`` akan mendapat jawaban "harga rumah $12" dengan status sukses.

    ``signature`` membuat kontrak itu ikut tersimpan: nama kolom + tipe. Kirim
    field yang salah, MLflow yang menolak lebih dulu — bukan model yang
    menebak-nebak.
    """
    if not _ada_run():
        return None

    prediksi = serving_model.predict(contoh_input)
    signature = infer_signature(contoh_input, prediksi)

    info = mlflow.sklearn.log_model(
        sk_model=serving_model,
        name=nama_artifact,
        signature=signature,
        input_example=contoh_input,
        # cloudpickle, bukan skops (bawaan MLflow 3): skops punya daftar tipe
        # tepercaya dan menolak XGBRegressor + FunctionTransformer kustom kita.
        serialization_format="cloudpickle",
    )
    logger.info("model_dicatat", extra={"model_uri": info.model_uri})
    return info.model_uri


def arahkan_tracking() -> None:
    """Arahkan MLflow ke database yang benar — TANPA menulis apa pun.

    Beda dengan ``setup()``: fungsi itu memanggil ``set_experiment`` yang MENULIS.
    Untuk sekadar membaca registry, menulis itu tidak perlu dan berbahaya —
    pada store SQLite di filesystem yang tidak mengizinkan hapus berkas,
    tulisan yang gagal meninggalkan berkas ``-journal`` yang membuat database
    tidak bisa dibuka sama sekali sesudahnya.
    """
    if not MLFLOW_TERSEDIA:
        raise RuntimeError("mlflow tidak terpasang")
    mlflow.set_tracking_uri(resolve_tracking_uri())


def _nama_registry(nama_model: str | None = None) -> str:
    return nama_model or config.get_settings().mlflow_model_name


def _champion_sekarang(client: Any, nama_model: str) -> tuple[Any | None, float | None]:
    """Ambil versi yang sedang memegang alias @champion beserta skor CV-nya."""
    try:
        versi = client.get_model_version_by_alias(nama_model, ALIAS_CHAMPION)
    except MlflowException:
        return None, None
    skor = versi.tags.get("cv_rmse_log")
    return versi, (float(skor) if skor is not None else None)


# ---------------------------------------------------------------------------
# Keputusan promosi — sengaja fungsi MURNI, tanpa MLflow sama sekali.
#
# Dipisah begini supaya aturannya bisa diuji tanpa menyalakan server, tanpa
# database, dan tanpa melatih model. Aturan promosi adalah bagian paling
# berisiko dari seluruh sistem — dia yang memutuskan model mana yang melayani
# orang — jadi dia yang paling pantas punya test sendiri.
# ---------------------------------------------------------------------------
def keputusan_promosi(
    skor_baru: float,
    std_baru: float,
    skor_lama: float | None,
    kontrak_baru: str | None = None,
    kontrak_lama: str | None = None,
    paksa: bool = False,
) -> dict[str, Any]:
    """Putuskan apakah challenger layak merebut alias @champion.

    Tiga saringan, berurutan:

    1. **Kontrak harus sama.** Model 79-fitur dan model 20-fitur menjawab soal
       yang berbeda; membandingkan RMSE keduanya sama saja membandingkan nilai
       ujian dari dua mata pelajaran.
    2. **Champion pertama menang otomatis** — tidak ada pembanding.
    3. **Margin kebisingan.** Challenger wajib menang lebih dari
       ``faktor x std CV``-nya sendiri. Menang 0.0001 pada model ber-std 0.015
       itu keberuntungan sampling, bukan perbaikan; mempromosikannya berarti
       menanggung risiko pergantian model tanpa dibayar apa pun.

    ``paksa=True`` melewati semuanya — dipakai untuk rollback sah, saat produksi
    bermasalah karena hal yang tidak terlihat di angka CV.
    """
    margin = max(float(std_baru or 0.0) * config.PROMOSI_MARGIN_STD_FAKTOR, 0.0)
    selisih = None if skor_lama is None else float(skor_lama) - float(skor_baru)

    if paksa:
        putusan, alasan = True, "dipaksa (margin & kontrak dilewati)"
    elif (config.PROMOSI_WAJIB_KONTRAK_SAMA and kontrak_lama is not None
          and kontrak_baru is not None and kontrak_baru != kontrak_lama):
        putusan, alasan = False, (f"kontrak berbeda ({kontrak_baru} vs champion "
                                  f"{kontrak_lama}) — angkanya tidak sebanding")
    elif skor_lama is None:
        putusan, alasan = True, "champion pertama"
    elif selisih > margin:
        putusan, alasan = True, f"menang {selisih:.5f} > margin {margin:.5f}"
    elif selisih > 0:
        putusan, alasan = False, (f"menang {selisih:.5f} tapi masih di dalam margin "
                                  f"kebisingan {margin:.5f} — belum terbukti lebih baik")
    else:
        putusan, alasan = False, f"kalah {abs(selisih):.5f} dari champion"

    return {"dipromosikan": putusan, "alasan": alasan, "margin": round(margin, 6),
            "selisih": None if selisih is None else round(selisih, 6)}


def promosikan(model_uri: str, metrics: dict[str, float], nama_model: str | None = None,
               paksa: bool = False, kontrak_hash: str | None = None) -> dict[str, Any] | None:
    """Daftarkan model ke registry, lalu putuskan apakah ia layak jadi champion.

    Pola champion–challenger: model baru SELALU didaftarkan (jejaknya disimpan
    dan bisa diperiksa nanti), tapi alias ``@champion`` hanya berpindah kalau
    ``keputusan_promosi`` meloloskannya.

    Kenapa CV yang dipakai membandingkan, bukan holdout? Karena holdout dibuka
    sekali saja untuk melapor. Begitu ia dipakai memilih antar-model, ia
    berhenti jadi ukuran jujur dan berubah jadi data validasi kedua.

    Yang berpindah cuma ALIAS. Versi lama tetap utuh di rak registry — itu yang
    membuat rollback jadi satu panggilan, bukan build ulang.
    """
    if not MLFLOW_TERSEDIA:
        return None

    nama_model = _nama_registry(nama_model)
    client = MlflowClient()

    versi_baru = mlflow.register_model(model_uri, nama_model)
    for kunci, nilai in metrics.items():
        if isinstance(nilai, (int, float)):
            client.set_model_version_tag(nama_model, versi_baru.version, kunci, f"{nilai:.5f}")
    client.set_model_version_tag(nama_model, versi_baru.version, "git_commit", git_commit())
    if kontrak_hash:
        client.set_model_version_tag(nama_model, versi_baru.version, "kontrak_hash", kontrak_hash)

    lama, skor_lama = _champion_sekarang(client, nama_model)
    keputusan = keputusan_promosi(
        skor_baru=float(metrics["cv_rmse_log"]),
        std_baru=float(metrics.get("cv_rmse_log_std", 0.0)),
        skor_lama=skor_lama,
        kontrak_baru=kontrak_hash,
        kontrak_lama=None if lama is None else lama.tags.get("kontrak_hash"),
        paksa=paksa,
    )

    if keputusan["dipromosikan"]:
        # Catat champion yang turun takhta SEBELUM alias dipindah — inilah yang
        # membuat rollback tahu harus kembali ke versi mana. Tanpa ini, "versi
        # sebelumnya" cuma bisa ditebak dari nomor urut, dan tebakan itu salah
        # begitu ada versi yang didaftarkan tapi tidak pernah jadi champion.
        if lama is not None:
            client.set_registered_model_tag(nama_model, "champion_sebelumnya", str(lama.version))
        client.set_registered_model_alias(nama_model, ALIAS_CHAMPION, versi_baru.version)

    hasil = {
        "nama_model": nama_model,
        "versi": int(versi_baru.version),
        "cv_rmse_log_baru": float(metrics["cv_rmse_log"]),
        "cv_rmse_log_champion_lama": skor_lama,
        "versi_champion_lama": int(lama.version) if lama is not None else None,
        "kontrak_hash": kontrak_hash,
        **keputusan,
    }
    logger.info("registry_diperbarui", extra=hasil)
    return hasil


def tetapkan_champion(versi: int | str, nama_model: str | None = None) -> dict[str, Any]:
    """Pindahkan alias @champion ke versi tertentu — inti dari seluruh pola ini.

    Yang berpindah HANYA penunjuk. Tidak ada model yang disalin, ditimpa, atau
    dilatih ulang; versi lama tetap duduk di rak registry. Itu sebabnya
    mengganti model produksi di sini berbiaya satu panggilan API, sementara
    kalau modelnya dipanggang ke image ia berbiaya build + deploy ulang.

    Champion yang turun takhta dicatat di tag ``champion_sebelumnya`` supaya
    rollback tahu harus kembali ke mana — bukan menebak dari nomor urut, yang
    salah begitu ada versi yang didaftarkan tapi tidak pernah jadi champion.
    """
    arahkan_tracking()
    nama_model = _nama_registry(nama_model)
    client = MlflowClient()

    try:
        client.get_model_version(nama_model, str(versi))
    except MlflowException as err:
        raise RuntimeError(f"versi {versi} tidak ada di '{nama_model}'") from err

    sekarang, _ = _champion_sekarang(client, nama_model)
    if sekarang is not None and str(sekarang.version) == str(versi):
        raise RuntimeError(f"v{versi} memang sudah memegang @{ALIAS_CHAMPION}")

    if sekarang is not None:
        client.set_registered_model_tag(nama_model, "champion_sebelumnya", str(sekarang.version))
    client.set_registered_model_alias(nama_model, ALIAS_CHAMPION, str(versi))

    hasil = {"nama_model": nama_model, "alias": ALIAS_CHAMPION,
             "dari_versi": None if sekarang is None else int(sekarang.version),
             "ke_versi": int(versi)}
    logger.warning("champion_dipindah", extra=hasil)
    return hasil


def rollback_champion(nama_model: str | None = None) -> dict[str, Any]:
    """Kembalikan alias @champion ke versi yang memegangnya sebelum promosi terakhir.

    Ini nilai jual utama pola alias: rollback tidak membangun ulang image, tidak
    melatih ulang, tidak menyalin berkas model. Ia memindahkan satu penunjuk.

    Memanggilnya dua kali akan bolak-balik antara dua versi — bukan menggali
    makin ke belakang. Itu disengaja: rollback adalah tindakan darurat untuk
    membatalkan SATU promosi, bukan mesin waktu.
    """
    arahkan_tracking()
    nama_model = _nama_registry(nama_model)
    client = MlflowClient()

    sekarang, _ = _champion_sekarang(client, nama_model)
    if sekarang is None:
        raise RuntimeError(f"'{nama_model}' belum punya champion — tidak ada yang dikembalikan")

    tujuan = client.get_registered_model(nama_model).tags.get("champion_sebelumnya")
    if not tujuan:
        raise RuntimeError("tidak ada catatan champion sebelumnya — rollback butuh "
                           "minimal satu kali pergantian champion")

    hasil = tetapkan_champion(tujuan, nama_model)
    logger.warning("champion_dikembalikan", extra=hasil)
    return hasil


def status_registry(nama_model: str | None = None) -> dict[str, Any]:
    """Isi rak registry: semua versi, metriknya, dan siapa yang memegang alias."""
    arahkan_tracking()
    nama_model = _nama_registry(nama_model)
    client = MlflowClient()

    sekarang, _ = _champion_sekarang(client, nama_model)
    try:
        model_terdaftar = client.get_registered_model(nama_model)
        sebelumnya = model_terdaftar.tags.get("champion_sebelumnya")
    except MlflowException:
        return {"nama_model": nama_model, "versi": [], "champion": None,
                "champion_sebelumnya": None, "catatan": "model belum terdaftar"}

    versi = []
    for v in client.search_model_versions(f"name='{nama_model}'"):
        versi.append({
            "versi": int(v.version),
            "champion": sekarang is not None and v.version == sekarang.version,
            "cv_rmse_log": v.tags.get("cv_rmse_log"),
            "holdout_rmse_log": v.tags.get("holdout_rmse_log"),
            "kontrak_hash": v.tags.get("kontrak_hash"),
            "git_commit": v.tags.get("git_commit"),
        })
    versi.sort(key=lambda x: x["versi"])
    return {"nama_model": nama_model, "versi": versi,
            "champion": None if sekarang is None else int(sekarang.version),
            "champion_sebelumnya": int(sebelumnya) if sebelumnya else None}


def versi_champion(nama_model: str | None = None) -> str:
    """Nomor versi yang sedang dipegang alias @champion — sekadar ANGKA-nya.

    Ini fungsi termurah di modul ini, dan itu disengaja: ia dipanggil berulang
    oleh pemantau pergantian model. Yang dilakukannya cuma satu query ke
    database registry — TIDAK mengunduh artifact, tidak memuat model.

    Kalau pemeriksaan berkala ikut mengunduh model, memantau jadi lebih mahal
    daripada melayani, dan MLflow yang dibebani tiap menit akan jadi titik
    lemah baru. Yang mahal (mengunduh + memuat) hanya dilakukan SETELAH
    terbukti versinya memang berbeda.
    """
    if not MLFLOW_TERSEDIA:
        raise RuntimeError("mlflow tidak terpasang")
    arahkan_tracking()
    nama_model = _nama_registry(nama_model)
    return str(MlflowClient().get_model_version_by_alias(nama_model, ALIAS_CHAMPION).version)


def muat_bundle_champion(nama_model: str | None = None) -> dict[str, Any]:
    """Muat BUNDLE produksi milik versi yang sedang memegang alias @champion.

    Yang diambil bukan model sklearn telanjang, melainkan artifact
    ``house_price_champion.joblib`` yang ikut dicatat di run milik versi itu —
    isinya sama persis dengan yang dipakai jalur ``MODEL_SOURCE=file``:
    preprocessor, model, kontrak API, dan metriknya.

    Kenapa begitu dan bukan ``mlflow.sklearn.load_model``? Supaya kedua jalur
    memuat BENDA YANG SAMA. Kalau jalur registry mengembalikan bentuk yang
    berbeda dari jalur file, kita baru saja membuat dua perilaku produksi yang
    harus dijaga sinkron selamanya — dan itu persis jenis perbedaan yang tidak
    memunculkan error, cuma angka yang berbeda.
    """
    if not MLFLOW_TERSEDIA:
        raise RuntimeError("mlflow tidak terpasang, tidak bisa memuat dari registry")

    import joblib

    arahkan_tracking()
    nama_model = _nama_registry(nama_model)
    versi = MlflowClient().get_model_version_by_alias(nama_model, ALIAS_CHAMPION)

    path = mlflow.artifacts.download_artifacts(
        run_id=versi.run_id,
        artifact_path=f"artifact_produksi/{config.MODEL_ARTIFACT.name}",
    )
    bundle = joblib.load(path)
    bundle["sumber_model"] = {
        "sumber": "registry", "nama_model": nama_model,
        "versi": int(versi.version), "alias": ALIAS_CHAMPION, "run_id": versi.run_id,
    }
    logger.info("model_dimuat_dari_registry",
                extra={"nama_model": nama_model, "versi": int(versi.version)})
    return bundle


def muat_champion(nama_model: str | None = None) -> Any:
    """Muat model sklearn end-to-end milik champion (masuk DataFrame, keluar USD).

    Dipakai kalau yang dibutuhkan objek model-nya langsung, bukan bundle —
    misalnya untuk analisis di notebook. Jalur serving memakai
    ``muat_bundle_champion`` supaya bentuknya sama dengan jalur file.
    """
    if not MLFLOW_TERSEDIA:
        raise RuntimeError("mlflow tidak terpasang, tidak bisa memuat dari registry")
    arahkan_tracking()
    return mlflow.sklearn.load_model(f"models:/{_nama_registry(nama_model)}@{ALIAS_CHAMPION}")
