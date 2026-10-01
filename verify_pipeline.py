#!/usr/bin/env python3
"""
verify_pipeline.py
------------------
Verifikasi data-pipeline / error (Bagian 10 task). Tiga jenis error ditangani:

  A. LABEL MISMATCH  - label di CSV split tidak sama dengan folder asli gambar
                       (atau label di luar {Cat, Dog}).
  B. FILE RUSAK      - file di split yang 0 byte / terpotong / tidak bisa di-decode,
                       atau tidak ada di valid_files.csv.
  C. SHAPE MISMATCH  - gambar non-RGB (grayscale, RGBA, CMYK, palette) menghasilkan tensor
                       dengan jumlah channel salah jika konversi RGB tidak diterapkan.

Plus pemeriksaan integritas split: tidak ada overlap path/MD5, SHA-256 CSV & jumlah data
cocok dengan split_manifest.json.

Dua mode:
  python src/verify_pipeline.py            # periksa data ASLI -> results/verification/pipeline_check.md
  python src/verify_pipeline.py --demo     # fault injection pada dataset mini sintetis
                                           # -> results/verification/pipeline_verification_demo.md

Exit code 1 jika mode data asli menemukan masalah (cocok dipakai sebagai gerbang sebelum training).
Bagian utama tidak membutuhkan torch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASS_TO_IDX, DEFAULT_MODEL, RESULTS_DIR, SPLIT_DIR, SPLIT_MANIFEST

CLASSES = tuple(CLASS_TO_IDX)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_splits(split_dir: Path) -> dict[str, list[dict]]:
    return {s: read_csv(split_dir / f"{s}.csv") for s in SPLITS}


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return rows
    idx = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(idx)]


# --------------------------------------------------------------------------- #
# Pemeriksaan (masing-masing mengembalikan daftar masalah; kosong = lolos)
# --------------------------------------------------------------------------- #
def check_labels(splits: dict[str, list[dict]]) -> list[str]:
    """A. Label harus valid dan sama dengan nama folder induk file."""
    issues = []
    for s, rows in splits.items():
        for r in rows:
            folder = Path(r["path"]).parent.name
            if r["label"] not in CLASS_TO_IDX:
                issues.append(f"[{s}] label tidak dikenal '{r['label']}': {r['path']}")
            elif r["label"] != folder:
                issues.append(f"[{s}] LABEL MISMATCH label='{r['label']}' tetapi folder='{folder}': {r['path']}")
    return issues


def check_overlap(splits: dict[str, list[dict]]) -> list[str]:
    issues = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("path", "md5"):
            if key not in (splits[a][0] if splits[a] else {}):
                continue
            inter = {r[key] for r in splits[a]} & {r[key] for r in splits[b]}
            if inter:
                issues.append(f"overlap {key} antara {a} dan {b}: {len(inter)} item")
    return issues


def check_manifest(split_dir: Path, manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return [f"manifest tidak ada: {manifest_path}"]
    mf = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    for s in SPLITS:
        p = split_dir / f"{s}.csv"
        if sha256_file(p) != mf["split_csv_sha256"][s]:
            issues.append(f"SHA-256 {s}.csv berbeda dari manifest (file berubah setelah split)")
        n = len(read_csv(p))
        if n != mf["counts"][s]["total"]:
            issues.append(f"jumlah baris {s}.csv = {n}, manifest = {mf['counts'][s]['total']}")
    return issues


def check_membership(splits: dict[str, list[dict]], valid_csv: Path | None, corrupt_csv: Path | None) -> list[str]:
    """B. Semua file split harus ada di valid_files.csv dan tidak ada di corrupt_files.csv."""
    issues = []
    if valid_csv and valid_csv.exists():
        valid = {r["path"] for r in read_csv(valid_csv)}
        for s, rows in splits.items():
            bad = [r["path"] for r in rows if r["path"] not in valid]
            if bad:
                issues.append(f"[{s}] {len(bad)} file tidak ada di valid_files.csv (contoh: {bad[0]})")
    if corrupt_csv and corrupt_csv.exists():
        corrupt = {r["path"] for r in read_csv(corrupt_csv)}
        for s, rows in splits.items():
            bad = [r["path"] for r in rows if r["path"] in corrupt]
            if bad:
                issues.append(f"[{s}] {len(bad)} file rusak ikut di split (contoh: {bad[0]})")
    return issues


def check_decodable(splits: dict[str, list[dict]], n: int, seed: int) -> list[str]:
    """B. Decode penuh (bukan hanya header) pada sampel; file 0 byte & terpotong tertangkap."""
    issues = []
    for s, rows in splits.items():
        for r in sample_rows(rows, n, seed):
            p = Path(r["path"])
            try:
                if p.stat().st_size == 0:
                    issues.append(f"[{s}] file kosong (0 byte): {r['path']}")
                    continue
                with Image.open(p) as im:
                    im.load()
            except Exception as e:  # noqa: BLE001
                issues.append(f"[{s}] tidak bisa di-decode ({type(e).__name__}): {r['path']}")
    return issues


def preprocess_numpy(path: str, size: int, force_rgb: bool) -> np.ndarray:
    """Replika ringkas pipeline: (konversi RGB opsional) -> resize -> array HxWxC."""
    with Image.open(path) as im:
        if force_rgb:
            im = im.convert("RGB")
        return np.asarray(im.resize((size, size), Image.BILINEAR))


def check_shapes(splits: dict[str, list[dict]], size: int, n: int, seed: int, force_rgb: bool = True) -> list[str]:
    """C. Setiap gambar harus menjadi (size, size, 3)."""
    issues = []
    expected = (size, size, DEFAULT_MODEL.in_channels)
    for s, rows in splits.items():
        for r in sample_rows(rows, n, seed):
            try:
                shape = preprocess_numpy(r["path"], size, force_rgb).shape
            except Exception as e:  # noqa: BLE001
                issues.append(f"[{s}] gagal preprocess ({type(e).__name__}): {r['path']}")
                continue
            if shape != expected:
                issues.append(f"[{s}] SHAPE MISMATCH {shape} != {expected}: {r['path']}")
    return issues


def run_all(split_dir: Path, manifest: Path, valid_csv: Path | None, corrupt_csv: Path | None,
            size: int, n: int, seed: int) -> dict[str, list[str]]:
    splits = load_splits(split_dir)
    return {
        "A. Label cocok dengan folder & valid": check_labels(splits),
        "Split: tanpa overlap path/MD5": check_overlap(splits),
        "Split: SHA-256 & jumlah cocok manifest": check_manifest(split_dir, manifest),
        "B. Semua file ada di valid_files.csv & tidak ada yang korup": check_membership(splits, valid_csv, corrupt_csv),
        "B. File dapat di-decode penuh (sampel)": check_decodable(splits, n, seed),
        f"C. Shape akhir = ({size},{size},3) (sampel)": check_shapes(splits, size, n, seed, True),
    }


# --------------------------------------------------------------------------- #
# Dataset mini untuk demo fault injection
# --------------------------------------------------------------------------- #
def build_demo_dataset(root: Path, per_class: int = 24, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASSES:
        d = root / "raw" / cls
        d.mkdir(parents=True)
        for i in range(per_class):
            h, w = int(rng.integers(90, 200)), int(rng.integers(90, 200))
            arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            im = Image.fromarray(arr)
            if i == 0:                      # satu gambar GRAYSCALE per kelas (seperti di dataset asli)
                im = im.convert("L")
            p = d / f"{i}.jpg"
            im.save(p)
            rows.append({"path": str(p), "label": cls, "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    # split stratified sederhana 70/15/15 per kelas (deterministik)
    splits = {s: [] for s in SPLITS}
    for cls in CLASSES:
        cr = [r for r in rows if r["label"] == cls]
        n_va, n_te = round(per_class * 0.15), round(per_class * 0.15)
        splits["test"] += cr[:n_te]
        splits["val"] += cr[n_te:n_te + n_va]
        splits["train"] += cr[n_te + n_va:]
    split_dir = root / "splits"
    split_dir.mkdir()
    for s in SPLITS:
        write_split(split_dir / f"{s}.csv", splits[s])
    write_manifest(split_dir, splits)
    vc = root / "valid_files.csv"
    with vc.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "md5"])
        w.writeheader()
        w.writerows(rows)
    return {"split_dir": split_dir, "manifest": split_dir / "split_manifest.json", "valid_csv": vc,
            "corrupt_csv": root / "corrupt_files.csv", "root": root}


def write_split(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "md5"])
        for r in rows:
            w.writerow([r["path"], r["label"], r["md5"]])


def write_manifest(split_dir: Path, splits: dict) -> None:
    mf = {"split_csv_sha256": {s: sha256_file(split_dir / f"{s}.csv") for s in SPLITS},
          "counts": {s: {"total": len(splits[s])} for s in SPLITS}}
    (split_dir / "split_manifest.json").write_text(json.dumps(mf, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Demo: suntik error -> deteksi -> perbaiki -> verifikasi ulang
# --------------------------------------------------------------------------- #
def run_demo(out_dir: Path, size: int, seed: int) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_demo_"))
    results = []  # (nama, sebelum_perbaikan, setelah_perbaikan, detail)
    try:
        ds = build_demo_dataset(tmp)
        sd, mf, vc, cc = ds["split_dir"], ds["manifest"], ds["valid_csv"], ds["corrupt_csv"]
        baseline = run_all(sd, mf, vc, cc, size, 0, seed)
        base_issues = sum(len(v) for v in baseline.values())

        # ---------- A. label mismatch ----------
        splits = load_splits(sd)
        victim = splits["train"][0]
        original_label = victim["label"]
        swapped_label = "Dog" if original_label == "Cat" else "Cat"
        victim["label"] = swapped_label                                     # injeksi: label tertukar
        write_split(sd / "train.csv", splits["train"])
        detect_a = check_labels(load_splits(sd))
        detect_a_manifest = check_manifest(sd, mf)                          # SHA-256 ikut berubah
        victim["label"] = Path(victim["path"]).parent.name                  # perbaikan: label dari folder
        write_split(sd / "train.csv", splits["train"])
        write_manifest(sd, splits)                                          # (pada pipeline asli: ulang split_data.py)
        fixed_a = check_labels(load_splits(sd)) + check_manifest(sd, mf)
        results.append(("A. Label mismatch", detect_a + detect_a_manifest, fixed_a,
                        f"label baris train[0] ditukar {original_label}->{swapped_label}"))

        # ---------- B. file rusak ----------
        splits = load_splits(sd)
        bad_row = splits["val"][0]
        bad_path = Path(bad_row["path"])
        raw = bad_path.read_bytes()
        backup = raw
        bad_path.write_bytes(raw[: len(raw) // 3])                          # injeksi: file terpotong
        zero_row = splits["val"][1]
        zero_backup = Path(zero_row["path"]).read_bytes()
        Path(zero_row["path"]).write_bytes(b"")                             # injeksi: file 0 byte
        detect_b = check_decodable(splits, 0, seed)
        with cc.open("w", newline="", encoding="utf-8") as f:               # inspect_dataset mencatat file rusak
            w = csv.writer(f)
            w.writerow(["path", "label", "size_bytes", "reason"])
            for r in (bad_row, zero_row):
                w.writerow([r["path"], r["label"], Path(r["path"]).stat().st_size, "decode gagal"])
        detect_b_member = check_membership(splits, vc, cc)
        # perbaikan: keluarkan file rusak dari split (pada pipeline asli: inspect_dataset -> split_data ulang)
        broken = {bad_row["path"], zero_row["path"]}
        splits["val"] = [r for r in splits["val"] if r["path"] not in broken]
        write_split(sd / "val.csv", splits["val"])
        write_manifest(sd, splits)
        fixed_b = check_decodable(load_splits(sd), 0, seed) + check_membership(load_splits(sd), vc, cc)
        bad_path.write_bytes(backup)
        Path(zero_row["path"]).write_bytes(zero_backup)
        results.append(("B. File rusak (terpotong + 0 byte)", detect_b + detect_b_member, fixed_b,
                        "1 file dipotong sepertiga, 1 file dikosongkan"))

        # ---------- C. shape mismatch ----------
        splits = load_splits(sd)
        detect_c = check_shapes(splits, size, 0, seed, force_rgb=False)     # pipeline NAIF tanpa convert("RGB")
        fixed_c = check_shapes(splits, size, 0, seed, force_rgb=True)       # pipeline final: convert("RGB")
        n_gray = sum(1 for v in splits.values() for r in v if Image.open(r["path"]).mode != "RGB")
        results.append(("C. Shape mismatch (grayscale)", detect_c, fixed_c,
                        f"{n_gray} gambar grayscale pada split; pipeline tanpa convert('RGB')"))

        path = write_demo_report(out_dir, results, base_issues, size)
        return path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_demo_report(out_dir: Path, results: list, base_issues: int, size: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Verifikasi Data-Pipeline: Demo Fault Injection\n",
         f"- Waktu (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         "- Dataset: mini sintetis (2 kelas x 24 gambar, termasuk 1 gambar grayscale per kelas); "
         "data asli tidak diubah.",
         f"- Garis dasar sebelum injeksi: **{base_issues} masalah** (pipeline bersih).\n",
         "## Ringkasan hasil\n",
         "| Error | Injeksi | Terdeteksi (jumlah temuan) | Setelah perbaikan |", "|---|---|---:|---:|"]
    for name, detected, after, note in results:
        L.append(f"| {name} | {note} | {len(detected)} | {len(after)} |")
    L += ["", "Temuan contoh yang tercetak oleh pemeriksa:\n"]
    for name, detected, _, _ in results:
        L.append(f"- **{name}**: `{detected[0][:150]}`" if detected else f"- **{name}**: (tidak terdeteksi!)")

    L += ["", "---", "", "## A. Label mismatch\n",
          "1. **Kemungkinan penyebab:** label dibuat dari urutan indeks yang tidak sinkron dengan daftar path "
          "(mis. `sort` hanya pada salah satu list), CSV diedit manual, atau pemetaan Cat/Dog tertukar "
          "antara script split dan training.",
          "2. **Gejala:** akurasi validasi aneh (jauh di bawah 50% bila tertukar sistematis, atau tidak pernah "
          "naik di atas ~50-70%), train loss turun tetapi val loss tidak, confusion matrix sangat simetris, "
          "gambar kucing dengan label Dog saat divisualisasikan.",
          "3. **Cara verifikasi:** `check_labels` membandingkan label setiap baris dengan nama folder file; "
          "SHA-256 `train/val/test.csv` dibandingkan dengan `split_manifest.json`; tampilkan grid sampel "
          "(`sample_batch.png`) beserta labelnya.",
          "4. **Perbaikan:** jangan edit CSV manual; turunkan label hanya dari folder (`label = parent.name`) dan "
          "jalankan ulang `split_data.py` (seed sama) agar manifest/SHA-256 ikut diperbarui.",
          "5. **Memastikan tidak terulang:** `verify_pipeline.py` dijadikan gerbang sebelum training (exit code 1 jika "
          "ada temuan), `verify_split_integrity()` pada `train.py` menolak CSV yang berubah, dan encoding label "
          "Cat=0/Dog=1 terpusat di `config.py`.\n",
          "## B. File gambar rusak\n",
          "1. **Kemungkinan penyebab:** unduhan tidak lengkap, file 0 byte atau terpotong, ekstensi `.jpg` tetapi isi "
          "bukan gambar (dataset ini memang memuat beberapa file seperti itu), atau file non-gambar seperti `Thumbs.db`.",
          "2. **Gejala:** `UnidentifiedImageError`, `OSError: image file is truncated`, DataLoader crash di tengah "
          "epoch, atau (jika diabaikan) gambar kosong/hitam yang masuk ke training.",
          "3. **Cara verifikasi:** decode penuh `im.load()` (bukan hanya `verify()`/header), cek ukuran 0 byte, cocokkan "
          "terhadap `valid_files.csv` dan `corrupt_files.csv` dari `inspect_dataset.py`.",
          "4. **Perbaikan:** keluarkan file dari daftar valid (jangan dihapus dari disk mentah), jalankan ulang "
          "`inspect_dataset.py` lalu `split_data.py`. `CatsDogsDataset` sengaja melempar `RuntimeError` yang menyebut "
          "path file, bukan melewatinya diam-diam.",
          "5. **Memastikan tidak terulang:** split hanya dibuat dari `valid_files.csv`; `check_membership` dan "
          "`check_decodable` pada `verify_pipeline.py` menghasilkan 0 temuan.\n",
          "## C. Shape mismatch\n",
          "1. **Kemungkinan penyebab:** gambar grayscale (1 channel), RGBA/PNG (4 channel), CMYK, atau palette tidak "
          "dikonversi ke RGB sebelum `ToTensor`; atau ukuran resize tidak sama dengan `input_size` model.",
          f"2. **Gejala:** `RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input to have 3 "
          f"channels, but got 1 channels`, atau `default_collate` gagal karena tensor dalam satu batch berbeda "
          f"bentuk (mis. (1,{size},{size}) vs (3,{size},{size})).",
          f"3. **Cara verifikasi:** `check_shapes` memastikan setiap gambar menjadi ({size},{size},3); "
          "`sanity_check_batch()` pada `preprocessing.py` meng-assert shape `(N,3,H,W)`, dtype, dan nilai label; "
          "`inspect_dataset.py` mencatat distribusi mode gambar (L/RGB/RGBA/CMYK).",
          "4. **Perbaikan:** paksa `im.convert('RGB')` di `CatsDogsDataset.__getitem__` dan `Resize((size, size))` yang "
          "sama untuk train/val/test.",
          "5. **Memastikan tidak terulang:** `check_shapes` dengan `force_rgb=True` memberi 0 temuan; assert di "
          "`train.py` (epoch pertama) dan `evaluate.py` gagal keras bila shape berubah.\n"]
    path = out_dir / "pipeline_verification_demo.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_check_report(out_dir: Path, results: dict[str, list[str]], args) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in results.values())
    L = ["# Hasil Pemeriksaan Data-Pipeline (data asli)\n",
         f"- Waktu (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         f"- Folder split: `{args.split_dir}` | sampel per split: {'semua' if args.sample <= 0 else args.sample} | seed {args.seed}",
         f"- Status: **{'LOLOS (0 masalah)' if total == 0 else f'{total} MASALAH DITEMUKAN'}**\n",
         "| Pemeriksaan | Temuan | Status |", "|---|---:|:---:|"]
    for name, issues in results.items():
        L.append(f"| {name} | {len(issues)} | {'OK' if not issues else 'GAGAL'} |")
    for name, issues in results.items():
        if issues:
            L += ["", f"### {name}", ""] + [f"- {i}" for i in issues[:20]]
            if len(issues) > 20:
                L.append(f"- ... dan {len(issues) - 20} lainnya")
    path = out_dir / "pipeline_check.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Verifikasi data-pipeline (label, file rusak, shape).")
    ap.add_argument("--demo", action="store_true", help="Demo fault injection pada dataset mini sintetis")
    ap.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    ap.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    ap.add_argument("--valid-csv", type=Path, default=RESULTS_DIR / "inspection" / "valid_files.csv")
    ap.add_argument("--corrupt-csv", type=Path, default=RESULTS_DIR / "inspection" / "corrupt_files.csv")
    ap.add_argument("--size", type=int, default=DEFAULT_MODEL.input_size)
    ap.add_argument("--sample", type=int, default=2000, help="Sampel per split untuk decode/shape (<=0 = semua)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "verification")
    args = ap.parse_args()

    if args.demo:
        path = run_demo(args.out_dir, args.size, args.seed)
        print(path.read_text(encoding="utf-8").split("\n---\n")[0])
        print(f"Laporan lengkap: {path}")
        return

    for p in [args.split_dir / f"{s}.csv" for s in SPLITS]:
        if not p.exists():
            sys.exit(f"[ERROR] {p} tidak ada. Jalankan dulu: python src/split_data.py")
    results = run_all(args.split_dir, args.manifest, args.valid_csv, args.corrupt_csv,
                      args.size, args.sample, args.seed)
    report = write_check_report(args.out_dir, results, args)
    total = sum(len(v) for v in results.values())
    for name, issues in results.items():
        print(f"[{'OK ' if not issues else 'GAGAL'}] {name}: {len(issues)} temuan")
        for i in issues[:5]:
            print(f"        - {i}")
    print(f"\nLaporan: {report}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
