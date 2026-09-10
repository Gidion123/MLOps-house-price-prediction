"""app.py — antarmuka Streamlit. Klien HTTP biasa, bukan bagian dari backend.

File ini tidak mengimpor apa pun dari ``src/`` maupun ``api/``. Ia tidak tahu
model apa yang dipakai, tidak memuat artifact, dan tidak menghitung satu pun
angka. Semua yang ditampilkan datang dari jawaban HTTP backend.

Konsekuensinya yang membuat pemisahan ini nyata: kalau besok backend dipindah
ke server lain, yang berubah cuma satu variabel lingkungan ``API_URL``.

Jalankan lokal (backend harus sudah hidup di :8000)::

    streamlit run frontend/app.py

Atau lewat compose, sudah termasuk backend & monitoring::

    docker compose up -d
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from frontend import kontrak

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------------------
# Pengambilan kontrak — di-cache supaya tidak menembak backend tiap interaksi
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def muat_kontrak():
    spec = kontrak.ambil_spesifikasi()
    return (kontrak.kelompokkan(kontrak.bidang_dari_skema(spec)),
            kontrak.contoh_request(spec))


@st.cache_data(ttl=60, show_spinner=False)
def muat_info():
    return kontrak.ambil_info_model()


def tampilkan_backend_mati(err: Exception) -> None:
    st.error("Backend tidak bisa dihubungi.")
    st.caption(f"Alamat yang dicoba: `{kontrak.BASE_URL}`")
    st.markdown(
        "Yang biasanya jadi sebab:\n"
        "- API belum jalan — coba `docker compose up -d` atau "
        "`uvicorn api.main:app --port 8000`\n"
        "- `API_URL` salah. Di dalam Docker nilainya `http://api:8000`, "
        "dari laptop `http://localhost:8000`"
    )
    with st.expander("Pesan teknis"):
        st.code(str(err))


# ---------------------------------------------------------------------------
# Sidebar — identitas model yang sedang melayani
# ---------------------------------------------------------------------------
def sidebar() -> None:
    with st.sidebar:
        st.subheader("Model yang melayani")
        try:
            info = muat_info()
        except kontrak.BackendTidakTerjangkau:
            st.warning("Backend belum terjangkau.")
            st.caption(f"`{kontrak.BASE_URL}`")
            return

        st.metric("Holdout R²", f"{info['metrics'].get('holdout_r2', 0):.4f}")
        st.metric("Rata-rata meleset (MAE)",
                  f"${info['metrics'].get('holdout_mae_usd', 0):,.0f}")
        st.caption(
            f"**{info['model_name']}** ({info['model_type']})  \n"
            f"Dilatih pada {info['training_rows']:,} rumah  \n"
            f"Dibuat {info['created_at']}"
        )
        with st.expander("Catatan"):
            st.caption(info["metrics_note"])
            st.caption(
                "Kolom yang nilainya baru ada setelah rumah terjual "
                f"({', '.join(info['leakage_features_removed'])}) sengaja dibuang, "
                "karena tidak akan pernah ada saat kamu bertanya."
            )
        st.divider()
        st.caption(f"API: `{kontrak.BASE_URL}`")


# ---------------------------------------------------------------------------
# Tab 1 — satu rumah
# ---------------------------------------------------------------------------
def render_bidang(b: dict, awal: dict):
    """Pilih widget berdasarkan schema, bukan berdasarkan nama field."""
    nilai_awal = awal.get(b["nama"])
    bantuan = b["keterangan"] or None

    if b["pilihan"]:
        pilihan = list(b["pilihan"])
        idx = pilihan.index(nilai_awal) if nilai_awal in pilihan else 0
        return st.selectbox(b["nama"], pilihan, index=idx, help=bantuan)

    # Tipe widget mengikuti tipe di schema, bukan tebakan. Tiga field luas
    # (GrLivArea, TotalBsmtSF, LotArea) dideklarasikan `number` oleh backend —
    # memaksanya jadi integer di sini berarti form menolak 1710.5 padahal API
    # menerimanya. Kontrak yang dipersempit diam-diam oleh UI tetap kontrak
    # yang dilanggar.
    pecahan = b["tipe"] == "number"
    cast = float if pecahan else int
    minimum = cast(b["min"]) if b["min"] is not None else cast(0)
    maksimum = cast(b["max"]) if b["max"] is not None else cast(1_000_000)
    awal = cast(nilai_awal) if nilai_awal is not None else minimum
    return st.number_input(b["nama"], min_value=minimum, max_value=maksimum,
                           value=max(minimum, min(awal, maksimum)),
                           step=cast(1), help=bantuan)


def tab_satu_rumah(grup: dict, contoh: dict) -> None:
    st.caption(
        "Isian awal sudah diisi contoh dari backend — tekan tombolnya dulu untuk "
        "melihat hasil, baru ubah angkanya satu per satu."
    )
    with st.form("form_rumah"):
        payload: dict = {}
        for judul, bidang in grup.items():
            st.markdown(f"**{judul}**")
            kolom = st.columns(min(len(bidang), 4))
            for i, b in enumerate(bidang):
                with kolom[i % len(kolom)]:
                    payload[b["nama"]] = render_bidang(b, contoh)
        kirim = st.form_submit_button("Hitung perkiraan harga", type="primary",
                                      use_container_width=True)

    if not kirim:
        return

    try:
        berhasil, jawaban = kontrak.prediksi(payload)
    except kontrak.BackendTidakTerjangkau as err:
        tampilkan_backend_mati(err)
        return

    if not berhasil:
        st.error(jawaban.get("error", "Request ditolak backend"))
        rincian = jawaban.get("detail")
        if isinstance(rincian, list):
            st.dataframe(pd.DataFrame(rincian), use_container_width=True, hide_index=True)
        else:
            st.code(str(rincian))
        st.caption(f"request_id: `{jawaban.get('request_id', '-')}`")
        return

    iv = jawaban["interval"]
    st.success("Perkiraan berhasil dihitung")
    k1, k2, k3 = st.columns(3)
    k1.metric("Batas bawah", f"${iv['bawah']:,.0f}")
    k2.metric("Perkiraan harga", f"${jawaban['predicted_price_usd']:,.0f}")
    k3.metric("Batas atas", f"${iv['atas']:,.0f}")

    # Angka tunggal terbaca sebagai kepastian. Ini yang menahan pembacaan itu.
    st.info(
        f"Rentang di atas adalah interval **{iv['tingkat_keyakinan']:.0%}** — dihitung dari "
        "sebaran error model di data holdout. Ini perkiraan, bukan jaminan: rumah yang "
        "sangat mahal cenderung meleset lebih jauh daripada yang tergambar di sini."
    )
    st.caption(
        f"Model **{jawaban['model_name']}** ({jawaban['model_type']}) · "
        f"dilatih pada {jawaban['model_trained_on_rows']:,} rumah · "
        f"{jawaban['latency_ms']:.1f} ms · request_id `{jawaban['request_id']}`"
    )


# ---------------------------------------------------------------------------
# Tab 2 — banyak rumah dari CSV
# ---------------------------------------------------------------------------
def tab_massal(grup: dict, contoh: dict) -> None:
    kolom_wajib = [b["nama"] for bidang in grup.values() for b in bidang]

    st.caption("Unggah CSV berisi 20 kolom kontrak. Satu baris = satu rumah.")
    contoh_csv = pd.DataFrame([contoh]).to_csv(index=False).encode()
    st.download_button("Unduh template CSV", contoh_csv, "template_rumah.csv",
                       "text/csv", help="Berisi satu baris contoh dengan kolom yang benar")

    berkas = st.file_uploader("Berkas CSV", type=["csv"], label_visibility="collapsed")
    if berkas is None:
        return

    df = pd.read_csv(io.BytesIO(berkas.getvalue()))
    kurang = [k for k in kolom_wajib if k not in df.columns]
    berlebih = [k for k in df.columns if k not in kolom_wajib]
    if kurang or berlebih:
        st.error("Kolom CSV tidak cocok dengan kontrak API.")
        if kurang:
            st.write("Kurang:", kurang)
        if berlebih:
            st.write("Berlebih:", berlebih)
        return

    st.write(f"Terbaca **{len(df)}** baris.")
    if not st.button("Hitung semua", type="primary"):
        return

    # Backend membatasi 100 rumah per panggilan, jadi dikirim per potongan.
    # Batas itu memang disengaja di sisi server supaya satu request tidak
    # menahan worker terlalu lama — frontend menyesuaikan, bukan melawannya.
    baris = df[kolom_wajib].to_dict(orient="records")
    hasil, gagal = [], None
    bar = st.progress(0.0, text="Mengirim ke backend...")
    for i in range(0, len(baris), 100):
        potongan = baris[i:i + 100]
        try:
            berhasil, jawaban = kontrak.prediksi_batch(potongan)
        except kontrak.BackendTidakTerjangkau as err:
            tampilkan_backend_mati(err)
            return
        if not berhasil:
            gagal = jawaban
            break
        hasil.extend(jawaban["predictions"])
        bar.progress(min((i + 100) / len(baris), 1.0), text=f"{len(hasil)}/{len(baris)} selesai")
    bar.empty()

    if gagal is not None:
        st.error(gagal.get("error", "Request ditolak backend"))
        st.code(str(gagal.get("detail"))[:2000])
        return

    keluar = df.copy()
    keluar["perkiraan_usd"] = [h["predicted_price_usd"] for h in hasil]
    keluar["batas_bawah"] = [h["interval"]["bawah"] for h in hasil]
    keluar["batas_atas"] = [h["interval"]["atas"] for h in hasil]

    k1, k2, k3 = st.columns(3)
    k1.metric("Rumah dihitung", f"{len(keluar):,}")
    k2.metric("Median perkiraan", f"${keluar['perkiraan_usd'].median():,.0f}")
    k3.metric("Rentang", f"${keluar['perkiraan_usd'].min():,.0f} – "
                         f"${keluar['perkiraan_usd'].max():,.0f}")

    st.dataframe(keluar, use_container_width=True, hide_index=True)
    st.download_button("Unduh hasil (CSV)", keluar.to_csv(index=False).encode(),
                       "hasil_prediksi.csv", "text/csv", type="primary")


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🏠 Prediksi Harga Rumah")
    st.caption("Ames, Iowa · form ini dibangun langsung dari kontrak yang diterbitkan API")
    sidebar()

    try:
        grup, contoh = muat_kontrak()
    except kontrak.BackendTidakTerjangkau as err:
        tampilkan_backend_mati(err)
        st.stop()

    satu, massal = st.tabs(["Satu rumah", "Banyak rumah (CSV)"])
    with satu:
        tab_satu_rumah(grup, contoh)
    with massal:
        tab_massal(grup, contoh)


if __name__ == "__main__":
    main()
