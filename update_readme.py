#!/usr/bin/env python3
"""
update_readme.py
----------------
Mengisi blok "informasi dataset" di README.md secara otomatis dari manifest, supaya
URL, TANGGAL PENGUNDUHAN, dan hasil pembagian data di README selalu sama dengan data yang
benar-benar dipakai (bukan ditulis tangan).

Sumber:
  data/raw/download_manifest.json        (download_dataset.py)  -> tanggal unduh, versi, fingerprint
  results/inspection/dataset_summary.json (inspect_dataset.py)  -> jumlah valid / korup / non-gambar
  data/splits/split_manifest.json        (split_data.py)        -> seed, rasio, jumlah, SHA-256

Hanya teks di antara penanda berikut yang diganti; bagian README lain tidak disentuh:
  <!-- DATASET_INFO:START -->  ...  <!-- DATASET_INFO:END -->

Jika dataset tidak diunduh lewat download_dataset.py (mis. sudah ada di mesin), tanggal unduh
dapat diberikan manual:  python src/update_readme.py --download-date 2026-09-20

Contoh:
  python src/update_readme.py
  uv run main.py readme
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from config import RESULTS_DIR, SPLIT_MANIFEST

START = "<!-- DATASET_INFO:START -->"
END = "<!-- DATASET_INFO:END -->"
DATASET_URL = "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"
RAW_MANIFEST = Path("data/raw/download_manifest.json")
INSPECTION_SUMMARY = RESULTS_DIR / "inspection" / "dataset_summary.json"
MISSING = "_belum tersedia - jalankan langkah terkait, lalu `uv run main.py readme`_"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def format_download_date(dl: dict | None, override: str | None) -> str:
    if override:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", override):
            sys.exit("[ERROR] --download-date harus berformat YYYY-MM-DD, mis. 2026-09-20")
        return f"**{override}** (diisi manual)"
    if not dl or not dl.get("downloaded_at_utc"):
        return ("**BELUM TERCATAT** - jalankan `uv run main.py download`, atau jika dataset sudah ada "
                "di mesin isi manual dengan `uv run main.py readme --download-date YYYY-MM-DD`")
    iso = dl["downloaded_at_utc"]                       # 2026-09-20T08:00:00+00:00
    date, _, rest = iso.partition("T")
    text = f"**{date}** (pukul {rest[:8]} UTC)"
    checked = dl.get("last_checked_at_utc", "")
    if checked and checked != iso:
        text += f"; terakhir diverifikasi {checked.partition('T')[0]}"
    return text


def build_block(dl: dict | None, insp: dict | None, split: dict | None, date_override: str | None) -> str:
    rows = [("URL dataset", DATASET_URL),
            ("Tanggal pengunduhan", format_download_date(dl, date_override))]
    if dl:
        rows += [("Metode unduh", f"`{dl.get('downloader', 'kagglehub')}`"),
                 ("Versi dataset Kaggle", f"{dl.get('resolved_version')}"
                                          f"{' (dikunci --version)' if dl.get('requested_version') else ''}"),
                 ("Fingerprint daftar file (SHA-256)", f"`{dl.get('file_list_sha256', '-')}`"),
                 ("File mentah (Cat / Dog / total)",
                  f"{dl['files_per_class_raw']['Cat']} / {dl['files_per_class_raw']['Dog']} / {dl['total_files_raw']}")]
    else:
        rows.append(("Metode unduh / versi / fingerprint", MISSING))

    if insp:
        rows.append(("Hasil inspeksi (valid / korup / non-gambar)",
                     f"{insp['total_valid']} / {insp['total_corrupt']} / {insp['total_non_image']}"))
    else:
        rows.append(("Hasil inspeksi (valid / korup / non-gambar)", MISSING))

    if split:
        c, r = split["counts"], split["ratios_target"]
        rows += [("Seed pembagian data", f"**{split['random_seed']}**"),
                 ("Rasio target (train / val / test)", f"{r['train']:.0%} / {r['val']:.0%} / {r['test']:.0%}"),
                 ("Dikeluarkan saat split (label konflik pada duplikat)", f"{split['n_excluded_label_conflict']}"),
                 ("Kelompok duplikat yang dijaga satu split", f"{split['n_duplicate_groups_kept_together']}")]
        total = sum(c[s]["total"] for s in ("train", "val", "test"))
        for s in ("train", "val", "test"):
            rows.append((f"Jumlah {s}", f"{c[s]['total']} ({c[s]['total'] / total:.2%}); "
                                         f"Cat {c[s]['Cat']}, Dog {c[s]['Dog']}"))
        for s in ("train", "val", "test"):
            rows.append((f"SHA-256 {s}.csv", f"`{split['split_csv_sha256'][s]}`"))
    else:
        rows.append(("Hasil pembagian data", MISSING))

    L = ["> Blok ini dibuat otomatis oleh `uv run main.py readme` dari manifest di `data/` dan `results/`; "
         "jangan diedit manual.", "", "| Item | Nilai |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isi blok informasi dataset di README dari manifest.")
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--download-date", default=None, help="Tanggal unduh manual (YYYY-MM-DD) bila tidak ada manifest")
    args = ap.parse_args()

    if not args.readme.exists():
        sys.exit(f"[ERROR] {args.readme} tidak ada.")
    text = args.readme.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        sys.exit(f"[ERROR] README harus memuat tepat satu pasang penanda {START} ... {END}.")

    dl, insp, split = load_json(RAW_MANIFEST), load_json(INSPECTION_SUMMARY), load_json(SPLIT_MANIFEST)
    block = build_block(dl, insp, split, args.download_date)
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    args.readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")

    print(f"[OK] {args.readme} diperbarui.")
    for name, obj in (("download_manifest.json", dl), ("dataset_summary.json", insp), ("split_manifest.json", split)):
        print(f"  {'ditemukan' if obj else 'TIDAK ADA '} : {name}")
    if not dl and not args.download_date:
        print("  [PERHATIAN] Tanggal pengunduhan belum tercatat. Gunakan --download-date YYYY-MM-DD.")


if __name__ == "__main__":
    main()
