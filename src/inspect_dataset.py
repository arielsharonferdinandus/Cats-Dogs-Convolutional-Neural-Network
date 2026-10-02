#!/usr/bin/env python3
"""
inspect_dataset.py
------------------
Inspeksi dan validasi dataset Cats vs Dogs SEBELUM split/training.

Yang diperiksa:
  * struktur folder & jumlah total file
  * jumlah file per kelas (mentah, valid, korup, non-gambar)
  * format file (ekstensi vs format asli hasil decode)
  * dimensi gambar (width, height, aspect ratio) dan channel/mode
  * file rusak / tidak bisa dibaca (0 byte, tidak bisa dibuka, terpotong, dsb.)
  * duplikat persis (MD5) -> penting agar tidak terjadi kebocoran antar split

Output (default: results/inspection/):
  dataset_summary.json    ringkasan lengkap (mesin-baca)
  dataset_summary.md      ringkasan siap salin ke laporan
  valid_files.csv         daftar file valid -> HANYA ini yang boleh dipakai split
  corrupt_files.csv       daftar file rusak + alasan (tidak dipakai training)
  non_image_files.csv     file non-gambar (mis. Thumbs.db)
  class_distribution.png  grafik jumlah per kelas
  image_size_distribution.png  histogram width/height

Contoh:
  python src/inspect_dataset.py
  python src/inspect_dataset.py --raw-dir data/raw/PetImages --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("Cat", "Dog")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
# ekstensi -> nama format PIL yang diharapkan
EXT_TO_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
                 ".gif": "GIF", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF"}
TINY_SIDE = 32  # sisi terpendek di bawah ini ditandai 'tiny' (peringatan, bukan korup)


# --------------------------------------------------------------------------- #
# Pemeriksaan satu file (dijalankan di worker process)
# --------------------------------------------------------------------------- #
def check_image(item: tuple[str, str]) -> dict:
    path, label = item
    p = Path(path)
    rec = {"path": path, "label": label, "status": "valid", "reason": "",
           "size_bytes": 0, "width": 0, "height": 0, "mode": "", "channels": 0,
           "format": "", "ext_format_mismatch": False, "warnings": "", "md5": ""}
    try:
        rec["size_bytes"] = os.path.getsize(p)
    except OSError as e:
        rec.update(status="corrupt", reason=f"tidak bisa stat file: {e}")
        return rec

    if rec["size_bytes"] == 0:
        rec.update(status="corrupt", reason="file kosong (0 byte)")
        return rec

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # tahap 1: header + verifikasi struktur
            with Image.open(p) as im:
                rec["format"] = im.format or ""
                im.verify()
            # tahap 2: decode penuh (verify() tidak menangkap data terpotong)
            with Image.open(p) as im:
                im.load()
                rec["width"], rec["height"] = im.size
                rec["mode"] = im.mode
                rec["channels"] = len(im.getbands())
        except Exception as e:  # noqa: BLE001 - semua kegagalan baca = korup
            rec.update(status="corrupt", reason=f"{type(e).__name__}: {e}"[:200])
            rec["width"] = rec["height"] = rec["channels"] = 0
            return rec
    rec["warnings"] = " | ".join(sorted({str(w.message)[:80] for w in caught}))

    expected = EXT_TO_FORMAT.get(p.suffix.lower())
    rec["ext_format_mismatch"] = bool(expected and rec["format"] and expected != rec["format"])

    h = hashlib.md5()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    rec["md5"] = h.hexdigest()
    return rec


# --------------------------------------------------------------------------- #
# Struktur folder & koleksi file
# --------------------------------------------------------------------------- #
def locate_raw_dir(arg: Path | None) -> Path:
    if arg:
        return arg
    manifest = Path("data/raw/download_manifest.json")
    if manifest.exists():
        return Path(json.loads(manifest.read_text(encoding="utf-8"))["class_root"])
    return Path("data/raw/PetImages")


def describe_structure(raw_dir: Path) -> dict:
    """Ringkas struktur folder (kedalaman 1) + file di luar folder kelas."""
    entries = sorted(raw_dir.iterdir())
    dirs = {e.name: sum(1 for x in e.rglob("*") if x.is_file()) for e in entries if e.is_dir()}
    loose = [e.name for e in entries if e.is_file()]
    return {"root": str(raw_dir), "subfolders_file_counts": dirs, "files_at_root": loose}


def collect_files(raw_dir: Path):
    image_items, non_image = [], []
    ext_counter = {c: Counter() for c in CLASS_NAMES}
    for cls in CLASS_NAMES:
        cdir = raw_dir / cls
        if not cdir.is_dir():
            sys.exit(f"[ERROR] Folder kelas tidak ditemukan: {cdir}")
        for p in sorted(cdir.rglob("*")):
            if not p.is_file():
                continue
            ext_counter[cls][p.suffix.lower() or "(tanpa ekstensi)"] += 1
            if p.suffix.lower() in IMAGE_EXTS:
                image_items.append((str(p), cls))
            else:
                non_image.append({"path": str(p), "label": cls, "size_bytes": p.stat().st_size})
    return image_items, non_image, ext_counter


# --------------------------------------------------------------------------- #
# Statistik
# --------------------------------------------------------------------------- #
def stats_1d(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "max": float(a.max()), "mean": round(float(a.mean()), 2),
            "median": float(np.median(a)), "std": round(float(a.std()), 2),
            "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def size_stats(recs: list[dict]) -> dict:
    w = [r["width"] for r in recs]
    h = [r["height"] for r in recs]
    ar = [r["width"] / r["height"] for r in recs if r["height"] > 0]
    return {"n": len(recs), "width": stats_1d(w), "height": stats_1d(h),
            "aspect_ratio_w_over_h": stats_1d(ar),
            "tiny_images_lt_%dpx" % TINY_SIDE: sum(1 for r in recs if min(r["width"], r["height"]) < TINY_SIDE)}


def find_duplicates(valid: list[dict]) -> list[list[str]]:
    groups = defaultdict(list)
    for r in valid:
        groups[r["md5"]].append(f"{r['label']}/{Path(r['path']).name}")
    return [g for g in groups.values() if len(g) > 1]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_plots(out: Path, summary: dict, valid: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. jumlah per kelas
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(CLASS_NAMES))
    vals = [summary["per_class"][c]["valid"] for c in CLASS_NAMES]
    bad = [summary["per_class"][c]["corrupt"] for c in CLASS_NAMES]
    ax.bar(x - 0.2, vals, 0.4, label="valid")
    ax.bar(x + 0.2, bad, 0.4, label="korup")
    for i, (v, b) in enumerate(zip(vals, bad)):
        ax.text(i - 0.2, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, b, str(b), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("Jumlah gambar")
    ax.set_title("Distribusi kelas (valid vs korup)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. histogram dimensi
    if valid:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key in zip(axes, ("width", "height")):
            for cls in CLASS_NAMES:
                data = [r[key] for r in valid if r["label"] == cls]
                ax.hist(data, bins=50, alpha=0.6, label=cls)
            ax.set_xlabel(f"{key} (px)")
            ax.set_ylabel("Jumlah gambar")
            ax.set_title(f"Distribusi {key}")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "image_size_distribution.png", dpi=150)
        plt.close(fig)


def write_markdown(path: Path, s: dict) -> None:
    L = []
    L.append("# Ringkasan Dataset\n")
    L.append(f"- Sumber: {s['dataset_source']}")
    L.append(f"- Waktu inspeksi (UTC): {s['inspected_at_utc']}")
    L.append(f"- Root dataset: `{s['structure']['root']}`")
    L.append(f"- Total file ditemukan (semua jenis): **{s['total_files_all']}**")
    L.append(f"- Gambar valid: **{s['total_valid']}** | korup: **{s['total_corrupt']}** "
             f"| non-gambar: **{s['total_non_image']}**\n")

    L.append("## Struktur folder\n")
    L.append("```")
    L.append(f"{Path(s['structure']['root']).name}/")
    for d, n in s["structure"]["subfolders_file_counts"].items():
        L.append(f"├── {d}/   ({n} file)")
    L.append("```\n")

    L.append("## Jumlah per kelas\n")
    L.append("| Kelas | Total file | Valid | Korup | Non-gambar | Proporsi valid |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for c in CLASS_NAMES:
        pc = s["per_class"][c]
        prop = pc["valid"] / s["total_valid"] * 100 if s["total_valid"] else 0
        L.append(f"| {c} | {pc['total_files']} | {pc['valid']} | {pc['corrupt']} "
                 f"| {pc['non_image']} | {prop:.2f}% |")
    L.append(f"\nRasio kelas (Cat : Dog, valid) = {s['class_ratio_cat_to_dog']}\n")

    L.append("## Format file\n")
    L.append("| Kelas | Ekstensi | Format hasil decode | Mode (channel) |")
    L.append("|---|---|---|---|")
    for c in CLASS_NAMES:
        pc = s["per_class"][c]
        L.append(f"| {c} | {dict(pc['extensions'])} | {dict(pc['formats'])} | {dict(pc['modes'])} |")
    L.append(f"\nFile dengan ekstensi tidak cocok dengan format asli: **{s['ext_format_mismatch_count']}**\n")

    L.append("## Statistik ukuran gambar (hanya gambar valid)\n")
    L.append("| Cakupan | Metrik | Min | Max | Mean | Median | Std | P5 | P95 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for scope, st in [("Semua", s["size_stats_all"])] + [(c, s["per_class"][c]["size_stats"]) for c in CLASS_NAMES]:
        for metric in ("width", "height", "aspect_ratio_w_over_h"):
            m = st.get(metric, {})
            if m:
                L.append(f"| {scope} | {metric} | {m['min']:g} | {m['max']:g} | {m['mean']:g} | "
                         f"{m['median']:g} | {m['std']:g} | {m['p5']:g} | {m['p95']:g} |")
    L.append(f"\nChannel (jumlah band) pada gambar valid: {s['channel_counts']}\n")

    L.append("## File korup / tidak dapat dibaca\n")
    L.append(f"Jumlah: **{s['total_corrupt']}** (daftar lengkap: `corrupt_files.csv`)\n")
    for r in s["corrupt_examples"]:
        L.append(f"- `{r['label']}/{Path(r['path']).name}` - {r['reason']}")
    L.append("")

    L.append("## Peringatan lain\n")
    L.append(f"- Gambar dengan peringatan decoder (mis. EXIF rusak) tetapi tetap terbaca: {s['files_with_decoder_warnings']}")
    L.append(f"- Kelompok duplikat persis (MD5): {s['duplicate_groups']} "
             f"(total file ikut kelompok: {s['duplicate_files']})")
    L.append(f"- Gambar sangat kecil (< {TINY_SIDE}px): {s['tiny_images']}\n")

    L.append("## Keputusan untuk tahap berikutnya\n")
    L.append("- Hanya file pada `valid_files.csv` yang boleh dipakai untuk split/training/validation/test.")
    L.append("- Semua gambar akan dikonversi ke RGB 3 channel pada preprocessing (menangani grayscale/RGBA/palette).")
    L.append("- Duplikat persis sebaiknya dikelompokkan pada split agar tidak bocor antar train/val/test (kolom `md5`).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Inspeksi & validasi dataset Cats vs Dogs.")
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="Folder berisi Cat/ dan Dog/ (default: dibaca dari download_manifest.json)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/inspection"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    raw_dir = locate_raw_dir(args.raw_dir)
    if not raw_dir.exists():
        sys.exit(f"[ERROR] {raw_dir} tidak ada. Jalankan dulu: python src/download_dataset.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    structure = describe_structure(raw_dir)
    image_items, non_image, ext_counter = collect_files(raw_dir)
    print(f"[INFO] Root: {raw_dir}")
    print(f"[INFO] {len(image_items)} file berekstensi gambar, {len(non_image)} file non-gambar")
    print(f"[INFO] Memeriksa dengan {args.workers} worker ...")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(check_image, image_items, chunksize=64), 1):
            records.append(rec)
            if i % 2500 == 0 or i == len(image_items):
                print(f"  {i}/{len(image_items)}", flush=True)

    valid = [r for r in records if r["status"] == "valid"]
    corrupt = [r for r in records if r["status"] == "corrupt"]

    per_class = {}
    for c in CLASS_NAMES:
        v = [r for r in valid if r["label"] == c]
        per_class[c] = {
            "total_files": sum(ext_counter[c].values()),
            "valid": len(v),
            "corrupt": sum(1 for r in corrupt if r["label"] == c),
            "non_image": sum(1 for r in non_image if r["label"] == c),
            "extensions": dict(ext_counter[c]),
            "formats": dict(Counter(r["format"] for r in v)),
            "modes": dict(Counter(f"{r['mode']} ({r['channels']}ch)" for r in v)),
            "size_stats": size_stats(v),
        }

    dup_groups = find_duplicates(valid)
    n_cat, n_dog = per_class["Cat"]["valid"], per_class["Dog"]["valid"]
    manifest = Path("data/raw/download_manifest.json")
    source = "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"
    if manifest.exists():
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        source += f" (versi {mf.get('resolved_version')}, fingerprint {str(mf.get('file_list_sha256'))[:12]}...)"

    summary = {
        "dataset_source": source,
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "structure": structure,
        "total_files_all": sum(pc["total_files"] for pc in per_class.values()),
        "total_valid": len(valid),
        "total_corrupt": len(corrupt),
        "total_non_image": len(non_image),
        "per_class": per_class,
        "class_ratio_cat_to_dog": f"{n_cat / n_dog:.4f} : 1" if n_dog else "n/a",
        "size_stats_all": size_stats(valid),
        "channel_counts": dict(Counter(r["channels"] for r in valid)),
        "ext_format_mismatch_count": sum(1 for r in valid if r["ext_format_mismatch"]),
        "files_with_decoder_warnings": sum(1 for r in valid if r["warnings"]),
        "tiny_images": sum(1 for r in valid if min(r["width"], r["height"]) < TINY_SIDE),
        "duplicate_groups": len(dup_groups),
        "duplicate_files": sum(len(g) for g in dup_groups),
        "duplicate_group_examples": dup_groups[:10],
        "corrupt_examples": [{k: r[k] for k in ("path", "label", "reason")} for r in corrupt[:15]],
    }

    out = args.out_dir
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out / "dataset_summary.md", summary)
    write_csv(out / "valid_files.csv", valid,
              ["path", "label", "width", "height", "mode", "channels", "format", "size_bytes", "md5"])
    write_csv(out / "corrupt_files.csv", corrupt, ["path", "label", "size_bytes", "reason"])
    write_csv(out / "non_image_files.csv", non_image, ["path", "label", "size_bytes"])
    make_plots(out, summary, valid)

    print("\n=== RINGKASAN ===")
    for c in CLASS_NAMES:
        pc = per_class[c]
        print(f"{c}: total={pc['total_files']} valid={pc['valid']} korup={pc['corrupt']} non-gambar={pc['non_image']}")
    print(f"Total valid={len(valid)} | korup={len(corrupt)} | non-gambar={len(non_image)} | "
          f"duplikat (grup)={len(dup_groups)}")
    print(f"Output di: {out.resolve()}")


if __name__ == "__main__":
    main()
