#!/usr/bin/env python3
"""
download_dataset.py
-------------------
Mengunduh Microsoft Cats vs Dogs Dataset dari Kaggle menggunakan library
`kagglehub`, lalu memverifikasi struktur foldernya dan mencatat manifest
untuk reproducibility.

Dataset : https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

Aspek reproducibility:
  * Cache kagglehub diarahkan ke dalam folder project (data/kagglehub_cache),
    bukan ~/.cache, sehingga lokasi data selalu sama di setiap mesin.
  * Versi dataset dapat dikunci dengan --version N (dicatat di manifest).
  * Manifest menyimpan versi, jumlah file per kelas, dan fingerprint SHA-256
    dari daftar file (path relatif + ukuran) untuk mendeteksi perbedaan data.
  * Struktur akhir yang stabil: data/raw/PetImages/{Cat,Dog}/ (symlink/copy).

Autentikasi Kaggle (pilih salah satu):
  1. ~/.kaggle/kaggle.json  -> {"username": "...", "key": "..."}
  2. Env var KAGGLE_USERNAME dan KAGGLE_KEY
  3. Env var KAGGLE_API_TOKEN (access token format baru)
  4. kagglehub.login() secara interaktif (di-trigger oleh opsi --login)

Contoh:
  python src/download_dataset.py
  python src/download_dataset.py --version 1 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

OWNER_SLUG = "shaunthesheep/microsoft-catsvsdogs-dataset"
CLASS_NAMES = ("Cat", "Dog")


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def find_class_root(base: Path) -> Path | None:
    """Cari folder yang memuat subfolder Cat dan Dog (umumnya 'PetImages')."""
    for cand in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        if all((cand / c).is_dir() for c in CLASS_NAMES):
            return cand
    return None


def count_files(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file())


def file_list_fingerprint(class_root: Path) -> str:
    """SHA-256 dari daftar (path relatif, ukuran byte) semua file.
    Cepat (tanpa membaca isi file) tetapi cukup untuk mendeteksi perbedaan data."""
    h = hashlib.sha256()
    for cls in CLASS_NAMES:
        for p in sorted((class_root / cls).iterdir()):
            if p.is_file():
                h.update(f"{cls}/{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()


def parse_version(path: Path) -> int | None:
    m = re.search(r"versions[/\\](\d+)", str(path))
    return int(m.group(1)) if m else None


def materialize(class_root: Path, target: Path) -> Path:
    """Buat data/raw/PetImages yang stabil: symlink, fallback ke copy (mis. Windows)."""
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(class_root.resolve(), target_is_directory=True)
        print(f"[INFO] Symlink dibuat: {target} -> {class_root.resolve()}")
    except (OSError, NotImplementedError):
        print(f"[INFO] Symlink tidak didukung, menyalin data ke {target} ...")
        shutil.copytree(class_root, target)
    return target


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Download Microsoft Cats vs Dogs via kagglehub.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Folder dasar data (default: ./data)")
    parser.add_argument("--version", type=int, default=None,
                        help="Kunci versi dataset Kaggle (default: versi terbaru)")
    parser.add_argument("--force", action="store_true",
                        help="Paksa unduh ulang (abaikan cache)")
    parser.add_argument("--login", action="store_true",
                        help="Login interaktif via kagglehub.login()")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cache_dir = data_dir / "kagglehub_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # WAJIB diset SEBELUM import kagglehub agar cache berada di dalam project
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    try:
        import kagglehub
    except ImportError:
        sys.exit("[ERROR] kagglehub belum terpasang. Jalankan: pip install -r requirements.txt")

    if args.login:
        kagglehub.login()

    handle = OWNER_SLUG if args.version is None else f"{OWNER_SLUG}/versions/{args.version}"
    print(f"[INFO] Handle dataset : {handle}")
    print(f"[INFO] Cache kagglehub: {cache_dir}")

    try:
        downloaded = Path(kagglehub.dataset_download(handle, force_download=args.force))
    except Exception as exc:
        sys.exit(
            f"[ERROR] Gagal mengunduh dataset: {exc}\n"
            "  Periksa kredensial Kaggle:\n"
            "  - Buat token di https://www.kaggle.com/settings ('Create New Token'), simpan ke\n"
            "    ~/.kaggle/kaggle.json (chmod 600), ATAU set KAGGLE_USERNAME & KAGGLE_KEY,\n"
            "  - atau jalankan ulang dengan --login."
        )

    class_root = find_class_root(downloaded)
    if class_root is None:
        sys.exit(f"[ERROR] Folder 'Cat' dan 'Dog' tidak ditemukan di {downloaded}")

    per_class = {c: count_files(class_root / c) for c in CLASS_NAMES}
    if any(n == 0 for n in per_class.values()):
        sys.exit(f"[ERROR] Salah satu kelas kosong: {per_class}")

    stable_root = materialize(class_root, data_dir / "raw" / class_root.name)

    manifest = {
        "dataset": OWNER_SLUG,
        "url": f"https://www.kaggle.com/datasets/{OWNER_SLUG}",
        "downloader": f"kagglehub {getattr(kagglehub, '__version__', 'unknown')}",
        "requested_version": args.version,
        "resolved_version": parse_version(downloaded),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kagglehub_path": str(downloaded),
        "class_root": str(stable_root),
        "files_per_class_raw": per_class,
        "total_files_raw": sum(per_class.values()),
        "file_list_sha256": file_list_fingerprint(class_root),
    }
    manifest_path = data_dir / "raw" / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== RINGKASAN UNDUHAN ===")
    print(f"Versi dataset  : {manifest['resolved_version']}")
    print(f"Lokasi (stabil): {stable_root}")
    for cls, n in per_class.items():
        print(f"  {cls:<4}: {n:>6} file")
    print(f"Total file     : {manifest['total_files_raw']}")
    print(f"Fingerprint    : {manifest['file_list_sha256'][:16]}...")
    print(f"Manifest       : {manifest_path}")
    print("\nCatatan: jumlah di atas masih mentah (termasuk file non-gambar seperti "
          "Thumbs.db & gambar korup). Validasi dilakukan pada inspect_dataset.py.")


if __name__ == "__main__":
    main()
