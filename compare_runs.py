#!/usr/bin/env python3
"""
compare_runs.py
---------------
1. Merangkum semua eksperimen di results/runs/*/ dalam satu tabel, DIURUTKAN BERDASARKAN
   VALIDATION LOSS (bukan test). Test set tidak dibaca.
2. Dengan --select <run>, mengunci konfigurasi final ke results/final/selected_config.json:
   nama run, alasan, SHA-256 checkpoint, metrik validasi, dan waktu pemilihan.

Kunci ini adalah bukti workflow (Q6): evaluate.py --split test menolak berjalan jika
  - kunci tidak ada,
  - checkpoint yang dievaluasi berbeda dari yang dikunci (SHA-256), atau
  - test set sudah pernah dibuka SEBELUM kunci dibuat (pemilihan tidak sah).

Contoh:
  python src/compare_runs.py
  python src/compare_runs.py --select baseline --reason "val loss terendah di antara 4 run"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import FINAL_DIR, RESULTS_DIR, SELECTION_LOCK, SPLIT_MANIFEST, TEST_ACCESS_LOG


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for cfg_path in sorted(runs_dir.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("test_set_accessed"):
            sys.exit(f"[ERROR] Run '{cfg['run_name']}' menandai test_set_accessed=true. "
                     "Run ini tidak boleh dipakai untuk pemilihan model.")
        if "best_val_loss" not in cfg:  # run belum selesai
            continue
        rows.append({
            "run_name": cfg["run_name"], "optimizer": cfg["optimizer"],
            "learning_rate": cfg["learning_rate"], "batch_size": cfg["batch_size"],
            "augment": cfg["augment"], "weight_decay": cfg.get("weight_decay", 0.0),
            "params": cfg["trainable_params"], "epochs_run": cfg["epochs_run"],
            "best_epoch": cfg["best_epoch"], "best_val_loss": cfg["best_val_loss"],
            "best_val_acc": cfg["best_val_acc"], "train_time_s": cfg["training_time_seconds"],
            "device": cfg["device"], "seed": cfg["seed"],
        })
    return sorted(rows, key=lambda r: r["best_val_loss"])


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    L = ["# Perbandingan Eksperimen (diurutkan berdasarkan validation loss)\n",
         "Seluruh keputusan memakai validation set. Test set tidak diakses.\n",
         "| # | Run | Optimizer | LR | Batch | Augmentasi | Epoch (terbaik/dijalankan) | Val loss | Val acc | Waktu (s) |",
         "|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['run_name']} | {r['optimizer']} | {r['learning_rate']} | {r['batch_size']} | "
                 f"{'ya' if r['augment'] else 'tidak'} | {r['best_epoch']}/{r['epochs_run']} | "
                 f"{r['best_val_loss']:.4f} | {r['best_val_acc']:.4f} | {r['train_time_s']} |")
    (out_dir / "runs_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lock_selection(run_name: str, rows: list[dict], runs_dir: Path, reason: str, force: bool) -> Path:
    if TEST_ACCESS_LOG.exists() and not force:
        sys.exit(f"[ERROR] {TEST_ACCESS_LOG} sudah ada: test set pernah dibuka. "
                 "Memilih konfigurasi setelah melihat test tidak sah.")
    if SELECTION_LOCK.exists() and not force:
        sys.exit(f"[ERROR] {SELECTION_LOCK} sudah ada. Konfigurasi final sudah dikunci "
                 "(gunakan --force hanya jika Anda mengulang seluruh eksperimen).")
    chosen = next((r for r in rows if r["run_name"] == run_name), None)
    if chosen is None:
        sys.exit(f"[ERROR] Run '{run_name}' tidak ditemukan/selesai. Tersedia: {[r['run_name'] for r in rows]}")
    ckpt = runs_dir / run_name / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"[ERROR] {ckpt} tidak ada.")

    lock = {
        "selected_run": run_name,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_criterion": "validation loss terendah (validation set saja)",
        "reason": reason,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "val_metrics_at_selection": {"best_epoch": chosen["best_epoch"],
                                     "val_loss": chosen["best_val_loss"], "val_acc": chosen["best_val_acc"]},
        "n_runs_compared": len(rows),
        "runs_compared": [r["run_name"] for r in rows],
        "hyperparameters": {k: chosen[k] for k in ("optimizer", "learning_rate", "batch_size",
                                                   "augment", "weight_decay", "seed")},
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST) if SPLIT_MANIFEST.exists() else None,
        "test_set_accessed_before_selection": False,
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return SELECTION_LOCK


def main() -> None:
    ap = argparse.ArgumentParser(description="Bandingkan run (validation) & kunci konfigurasi final.")
    ap.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "runs")
    ap.add_argument("--select", metavar="RUN", help="Kunci run ini sebagai konfigurasi final")
    ap.add_argument("--reason", default="", help="Alasan pemilihan (dicatat di kunci)")
    ap.add_argument("--force", action="store_true", help="Timpa kunci lama (hanya jika mengulang eksperimen)")
    args = ap.parse_args()

    rows = load_runs(args.runs_dir)
    if not rows:
        sys.exit(f"[ERROR] Tidak ada run selesai di {args.runs_dir}. Jalankan python src/train.py dulu.")
    write_outputs(rows, RESULTS_DIR)

    print(f"{'#':>2} {'run':<16}{'opt':<7}{'lr':>8}{'bs':>5}{'aug':>5}{'best ep':>9}{'val loss':>10}{'val acc':>9}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2} {r['run_name']:<16}{r['optimizer']:<7}{r['learning_rate']:>8}{r['batch_size']:>5}"
              f"{str(r['augment'])[0]:>5}{r['best_epoch']:>9}{r['best_val_loss']:>10.4f}{r['best_val_acc']:>9.4f}")
    print(f"\nTabel: {RESULTS_DIR / 'runs_comparison.md'}")

    if args.select:
        path = lock_selection(args.select, rows, args.runs_dir, args.reason, args.force)
        print(f"[OK] Konfigurasi final DIKUNCI: run '{args.select}' -> {path}")
        print("     Langkah berikut: python src/evaluate.py --split test --confirm-final")
    elif rows[0]["run_name"]:
        print(f"Val loss terendah: '{rows[0]['run_name']}'. Kunci dengan: "
              f"python src/compare_runs.py --select {rows[0]['run_name']}")


if __name__ == "__main__":
    main()
