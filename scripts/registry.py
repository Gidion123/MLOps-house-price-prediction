"""registry.py — lihat isi rak model, pindahkan alias, batalkan promosi.

    python -m scripts.registry                      # status rak registry
    python -m scripts.registry --rollback           # batalkan promosi terakhir
    python -m scripts.registry --jadikan-champion 2 # pindah alias ke versi 2

Semua perintah di sini bekerja pada ALIAS, bukan pada berkas model. Tidak ada
model yang disalin atau ditimpa — yang berubah cuma penunjuk. Itu sebabnya
mengembalikan model produksi ke versi sebelumnya selesai dalam hitungan detik.

Setelah alias berpindah, container yang sedang jalan belum ikut berpindah
sampai cache modelnya dibuang::

    curl -X POST localhost:8000/admin/rollback -H "X-Admin-Token: ..."
    # atau
    curl -X POST localhost:8000/admin/reload   -H "X-Admin-Token: ..."
"""

from __future__ import annotations

import argparse
import sys

from src.utils import tracking


def cetak_status(nama_model: str | None = None) -> None:
    st = tracking.status_registry(nama_model)
    print("\n" + "=" * 78)
    print(f"MODEL REGISTRY — {st['nama_model']}")
    print("=" * 78)
    if not st["versi"]:
        print("  Belum ada versi terdaftar. Jalankan: python -m src.models.trainer\n")
        return

    print(f"  {'versi':<8}{'CV RMSE_log':>13}{'holdout':>11}{'kontrak':>11}{'commit':>11}   alias")
    print("-" * 78)
    for v in st["versi"]:
        alias = f"@{tracking.ALIAS_CHAMPION}" if v["champion"] else ""
        print(f"  v{str(v['versi']):<7}{str(v['cv_rmse_log'] or '-'):>13}"
              f"{str(v['holdout_rmse_log'] or '-'):>11}{str(v['kontrak_hash'] or '-'):>11}"
              f"{str(v['git_commit'] or '-'):>11}   {alias}")
    print("-" * 78)
    print(f"  champion sekarang   : v{st['champion']}")
    print(f"  champion sebelumnya : "
          f"{'v' + str(st['champion_sebelumnya']) if st['champion_sebelumnya'] else '(belum pernah berganti)'}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Kelola alias champion di MLflow Registry")
    p.add_argument("--rollback", action="store_true",
                   help="kembalikan @champion ke versi sebelum promosi terakhir")
    p.add_argument("--jadikan-champion", type=int, default=None, metavar="VERSI",
                   help="pindahkan @champion ke versi tertentu")
    p.add_argument("--model", default=None, help="nama model terdaftar (default dari .env)")
    args = p.parse_args()

    if args.rollback and args.jadikan_champion is not None:
        print("Pilih salah satu: --rollback ATAU --jadikan-champion", file=sys.stderr)
        raise SystemExit(2)

    try:
        if args.rollback:
            h = tracking.rollback_champion(args.model)
            print(f"\n  ROLLBACK: @{h['alias']} v{h['dari_versi']} -> v{h['ke_versi']}")
        elif args.jadikan_champion is not None:
            h = tracking.tetapkan_champion(args.jadikan_champion, args.model)
            print(f"\n  @{h['alias']} dipindah: v{h['dari_versi']} -> v{h['ke_versi']}")
    except RuntimeError as err:
        print(f"\n  GAGAL: {err}\n", file=sys.stderr)
        raise SystemExit(1)

    cetak_status(args.model)
    if args.rollback or args.jadikan_champion is not None:
        print("  Container yang sedang jalan belum ikut berpindah — panggil "
              "/admin/reload untuk memuat ulang.\n")


if __name__ == "__main__":
    main()
