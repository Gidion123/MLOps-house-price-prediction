"""kontrak.py — frontend membaca kontrak dari backend, tidak menyalinnya.

Ini bagian terpenting dari pemisahan frontend/backend, dan yang paling gampang
dilanggar tanpa sadar.

Cara yang salah: menyalin daftar 25 Neighborhood dan batas 1-10 OverallQual ke
dalam kode Streamlit. Kelihatan praktis. Lalu suatu hari backend dilatih ulang
dengan kategori baru, dan form di frontend masih menawarkan daftar lama —
user memilih nilai yang ditolak server, dan tidak ada satu pun yang salah
menurut kode masing-masing.

Cara di sini: form dibangun dari ``/openapi.json`` yang **diterbitkan backend
sendiri**. Batas, pilihan, dan keterangan tiap field berasal dari schema
Pydantic yang sama persis dengan yang memvalidasi request. Satu sumber
kebenaran, dua proses.

Modul ini sengaja tidak mengimpor Streamlit maupun apa pun dari ``src/`` —
supaya bisa diuji tanpa menyalakan UI, dan supaya kaidah "frontend tidak boleh
mengimpor backend" bisa ditegakkan test.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 10

# Pengelompokan ini murni urusan tampilan, bukan kontrak. Field yang tidak
# terdaftar di sini tetap muncul di grup "Lainnya" — jadi kalau backend suatu
# hari menambah field, form ikut menampilkannya tanpa kode ini disentuh.
KELOMPOK: dict[str, list[str]] = {
    "Kualitas & kondisi": ["OverallQual", "OverallCond", "ExterQual", "KitchenQual", "HeatingQC"],
    "Luas": ["GrLivArea", "TotalBsmtSF", "LotArea"],
    "Umur bangunan": ["YearBuilt", "YearRemodAdd"],
    "Kamar mandi": ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
    "Fasilitas lain": ["GarageCars", "Fireplaces", "BsmtQual", "CentralAir"],
    "Lokasi": ["Neighborhood", "MSZoning"],
}


class BackendTidakTerjangkau(Exception):
    """Backend tidak menjawab. Dibedakan dari error lain supaya UI bisa
    menampilkan pesan yang benar-benar menolong, bukan 'terjadi kesalahan'."""


def _ambil(path: str) -> dict[str, Any]:
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as err:
        raise BackendTidakTerjangkau(f"gagal menghubungi {BASE_URL}{path}: {err}") from err


def ambil_spesifikasi() -> dict[str, Any]:
    return _ambil("/openapi.json")


def ambil_info_model() -> dict[str, Any]:
    return _ambil("/model-info")


def cek_kesehatan() -> dict[str, Any]:
    return _ambil("/health")


def bidang_dari_skema(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Ubah schema OpenAPI jadi daftar deskripsi field siap-render.

    Yang diambil dari tiap properti: tipe, batas bawah/atas, daftar pilihan,
    dan keterangan. Persis bahan yang dibutuhkan untuk memilih antara
    ``number_input`` dan ``selectbox``, lengkap dengan batasnya.
    """
    skema = spec["components"]["schemas"]["HousePriceRequest"]
    wajib = set(skema.get("required", []))

    bidang = []
    for nama, p in skema["properties"].items():
        bidang.append({
            "nama": nama,
            "tipe": p.get("type", "string"),
            "pilihan": p.get("enum"),
            "min": p.get("minimum"),
            "max": p.get("maximum"),
            "keterangan": p.get("description", ""),
            "wajib": nama in wajib,
        })
    return bidang


def contoh_request(spec: dict[str, Any]) -> dict[str, Any]:
    """Contoh isian dari backend — dipakai sebagai nilai awal form.

    Form kosong memaksa user mengisi 20 field sebelum bisa melihat apa pun.
    Form yang sudah terisi contoh bisa langsung ditekan, lalu diubah satu-satu.
    """
    skema = spec["components"]["schemas"]["HousePriceRequest"]
    contoh = skema.get("examples") or []
    return dict(contoh[0]) if contoh else {}


def kelompokkan(bidang: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Susun field ke dalam kelompok tampilan; sisanya masuk 'Lainnya'."""
    per_nama = {b["nama"]: b for b in bidang}
    hasil: dict[str, list[dict[str, Any]]] = {}
    terpakai = set()

    for judul, daftar in KELOMPOK.items():
        isi = [per_nama[n] for n in daftar if n in per_nama]
        if isi:
            hasil[judul] = isi
            terpakai.update(b["nama"] for b in isi)

    sisa = [b for b in bidang if b["nama"] not in terpakai]
    if sisa:
        hasil["Lainnya"] = sisa
    return hasil


def prediksi(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Kirim satu request prediksi. Kembalikan (berhasil, isi jawaban).

    422 TIDAK dilempar sebagai error. Ia jawaban sah dari backend yang berisi
    field mana yang bermasalah — justru itu yang mau ditampilkan ke user.
    """
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException as err:
        raise BackendTidakTerjangkau(f"gagal menghubungi {BASE_URL}/predict: {err}") from err
    return r.status_code == 200, r.json()


def prediksi_batch(baris: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    try:
        r = requests.post(f"{BASE_URL}/predict/batch", json={"houses": baris}, timeout=60)
    except requests.RequestException as err:
        raise BackendTidakTerjangkau(f"gagal menghubungi {BASE_URL}/predict/batch: {err}") from err
    return r.status_code == 200, r.json()
