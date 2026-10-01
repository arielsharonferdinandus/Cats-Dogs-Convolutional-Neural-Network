#!/usr/bin/env python3
"""
split_data.py
-------------
Membagi dataset menjadi train / validation / test secara reproducible.

Prinsip:
  * Input HANYA results/inspection/valid_files.csv (file rusak tidak ikut).
  * Stratified per kelas: proporsi Cat/Dog di tiap split mengikuti proporsi asli.
  * Group-aware: file dengan MD5 identik (duplikat persis) selalu masuk split yang
    sama, agar tidak ada kebocoran (leakage) antara train/val/test.
  * Jika satu kelompok duplikat punya label berbeda (Cat vs Dog) -> label conflict,
    kelompok itu dikeluarkan dan dicatat.
  * Random seed dicatat di manifest; hasil split identik untuk seed yang sama.
  * Setelah split, dilakukan pemeriksaan otomatis: tidak ada overlap path/MD5
    antar split dan rasio sesuai target.

Output:
  data/splits/train.csv, val.csv, test.csv   (path,label,md5)
  data/splits/split_manifest.json            (seed, rasio, jumlah, SHA-256 tiap CSV)
  results/split/split_summary.md             (ringkasan untuk laporan)

Catatan isolasi test set: test.csv hanya boleh dibaca oleh evaluate.py dan
error_analysis.py. SHA-256 test.csv dicatat agar perubahan diam-diam terdeteksi.

Contoh:
  python src/split_data.py
  python src/split_data.py --seed 42 --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CLASS_NAMES = ("Cat", "Dog")
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_valid_files(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[ERROR] {path} tidak ada. Jalankan dulu: python src/inspect_dataset.py")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("[ERROR] valid_files.csv kosong.")
    return rows


def build_groups(rows: list[dict]):
    """Kelompokkan berdasarkan MD5. Kembalikan (grup_per_kelas, grup_konflik)."""
    by_md5 = defaultdict(list)
    for r in rows:
        by_md5[r["md5"]].append(r)

    per_class = {c: [] for c in CLASS_NAMES}
    conflicts = []
    for md5, members in by_md5.items():
        labels = {m["label"] for m in members}
        if len(labels) > 1:
            conflicts.append(members)
        else:
            per_class[members[0]["label"]].append(members)
    return per_class, conflicts


def allocate_class(groups: list[list[dict]], ratios: dict[str, float], rng) -> dict[str, list[dict]]:
    """Alokasikan grup satu kelas ke tiga split dengan target jumlah per split.
    Grup diacak (seed) lalu dimasukkan ke split dengan kekurangan relatif terbesar."""
    n_total = sum(len(g) for g in groups)
    target = {s: ratios[s] * n_total for s in SPLITS}
    filled = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}

    # Acak (seed) lalu urutkan grup besar lebih dulu agar duplikat tidak merusak
    # rasio di akhir. sorted() stabil, jadi urutan acak tetap dipertahankan
    # di dalam grup dengan ukuran yang sama -> deterministik untuk seed yang sama.
    shuffled = rng.permutation(len(groups)).tolist()
    final_order = sorted(shuffled, key=lambda i: -len(groups[i]))

    for i in final_order:
        g = groups[i]
        # kekurangan relatif: (target - terisi) / target
        deficit = {s: (target[s] - filled[s]) / target[s] if target[s] > 0 else -1 for s in SPLITS}
        best = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
        out[best].extend(g)
        filled[best] += len(g)
    return out


def write_split_csv(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["label"], r["path"]))  # urutan deterministik
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "md5"])
        for r in rows:
            w.writerow([r["path"], r["label"], r["md5"]])


def verify_split(splits: dict[str, list[dict]], ratios: dict[str, float], tol: float = 0.01) -> dict:
    """Pemeriksaan integritas split. Menghentikan program jika ada kebocoran."""
    checks = {}
    paths = {s: {r["path"] for r in rows} for s, rows in splits.items()}
    md5s = {s: {r["md5"] for r in rows} for s, rows in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        path_overlap = len(paths[a] & paths[b])
        md5_overlap = len(md5s[a] & md5s[b])
        checks[f"{a}_vs_{b}_path_overlap"] = path_overlap
        checks[f"{a}_vs_{b}_md5_overlap"] = md5_overlap
        if path_overlap or md5_overlap:
            sys.exit(f"[ERROR] Kebocoran data antara {a} dan {b}: {checks}")

    total = sum(len(v) for v in splits.values())
    for s in SPLITS:
        actual = len(splits[s]) / total
        checks[f"{s}_ratio_actual"] = round(actual, 4)
        if abs(actual - ratios[s]) > tol:
            sys.exit(f"[ERROR] Rasio {s} menyimpang: {actual:.4f} vs target {ratios[s]:.4f}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Split train/val/test reproducible.")
    ap.add_argument("--valid-csv", type=Path, default=Path("results/inspection/valid_files.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--report-dir", type=Path, default=Path("results/split"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        sys.exit(f"[ERROR] Jumlah rasio harus 1.0, sekarang {sum(ratios.values())}")

    rows = read_valid_files(args.valid_csv)
    per_class_groups, conflicts = build_groups(rows)
    n_conflict_files = sum(len(g) for g in conflicts)
    n_dup_groups = sum(1 for c in CLASS_NAMES for g in per_class_groups[c] if len(g) > 1)

    # satu RNG, satu urutan kelas tetap -> deterministik
    rng = np.random.default_rng(args.seed)
    splits = {s: [] for s in SPLITS}
    for cls in CLASS_NAMES:
        alloc = allocate_class(per_class_groups[cls], ratios, rng)
        for s in SPLITS:
            splits[s].extend(alloc[s])

    checks = verify_split(splits, ratios)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_hash = {}
    for s in SPLITS:
        p = args.out_dir / f"{s}.csv"
        write_split_csv(p, splits[s])
        csv_hash[s] = sha256_file(p)

    counts = {s: {c: sum(1 for r in splits[s] if r["label"] == c) for c in CLASS_NAMES} for s in SPLITS}
    for s in SPLITS:
        counts[s]["total"] = len(splits[s])

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": args.seed,
        "method": "stratified per kelas + group-aware (MD5 duplikat tidak dipisah), "
                  "numpy.random.default_rng(seed)",
        "ratios_target": ratios,
        "source_valid_csv": str(args.valid_csv),
        "source_valid_csv_sha256": sha256_file(args.valid_csv),
        "n_input_valid": len(rows),
        "n_excluded_label_conflict": n_conflict_files,
        "n_duplicate_groups_kept_together": n_dup_groups,
        "counts": counts,
        "integrity_checks": checks,
        "split_csv_sha256": csv_hash,
        "test_set_policy": "test.csv hanya dibaca oleh evaluate.py / error_analysis.py "
                           "setelah konfigurasi final dikunci.",
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ringkasan Markdown untuk laporan
    L = ["# Ringkasan Data Split\n",
         f"- Metode: {manifest['method']}",
         f"- Random seed: **{args.seed}**",
         f"- Rasio target: train {args.train:.0%} / val {args.val:.0%} / test {args.test:.0%}",
         f"- Gambar valid masuk split: {len(rows) - n_conflict_files} dari {len(rows)} "
         f"(dikeluarkan karena label konflik pada duplikat: {n_conflict_files})",
         f"- Kelompok duplikat yang dijaga tetap satu split: {n_dup_groups}\n",
         "| Split | Cat | Dog | Total | Proporsi aktual | Proporsi Cat |",
         "|---|---:|---:|---:|---:|---:|"]
    total = sum(counts[s]["total"] for s in SPLITS)
    for s in SPLITS:
        c = counts[s]
        L.append(f"| {s} | {c['Cat']} | {c['Dog']} | {c['total']} | "
                 f"{c['total'] / total:.2%} | {c['Cat'] / c['total']:.2%} |")
    L += ["", "## Pemeriksaan integritas", "",
          "- Overlap path antar split: " + str(sum(v for k, v in checks.items() if k.endswith("path_overlap"))),
          "- Overlap MD5 (konten identik) antar split: " + str(sum(v for k, v in checks.items() if k.endswith("md5_overlap"))),
          "", "## SHA-256 file split", ""]
    L += [f"- `{s}.csv`: `{csv_hash[s]}`" for s in SPLITS]
    (args.report_dir / "split_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("=== HASIL SPLIT ===")
    print(f"Seed: {args.seed} | rasio {args.train}/{args.val}/{args.test}")
    for s in SPLITS:
        c = counts[s]
        print(f"{s:<5}: total={c['total']:>6}  Cat={c['Cat']:>6}  Dog={c['Dog']:>6}")
    print(f"Dikeluarkan (label konflik): {n_conflict_files}")
    print("Pemeriksaan kebocoran: OK (0 overlap path/MD5)")
    print(f"Output: {args.out_dir.resolve()} dan {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
