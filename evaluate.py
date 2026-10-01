#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluasi checkpoint pada validation set ATAU test set (accuracy, precision, recall,
F1-score, confusion matrix, ROC-AUC).

  --split val   : boleh dijalankan kapan saja. Hasil -> results/runs/<run>/
  --split test  : EVALUASI AKHIR, hanya sekali. Hasil -> results/final/
                  Syarat (diperiksa otomatis SEBELUM satu pun gambar test dibaca):
                    1. Konfigurasi final sudah dikunci (compare_runs.py --select ...),
                    2. run yang dievaluasi = run yang dikunci, dan SHA-256 checkpoint
                       sama persis dengan saat dikunci,
                    3. test set belum pernah dievaluasi (log results/final/test_access_log.jsonl;
                       ulang hanya dengan --allow-rerun dan akan tercatat sebagai rerun),
                    4. flag --confirm-final diberikan secara eksplisit.
                  Setiap akses test dicatat (waktu, run, SHA-256 checkpoint & test.csv).

Threshold klasifikasi TETAP 0.5 (P(Dog) >= 0.5 -> Dog); tidak di-tuning pada test.
Positive class = Dog.

Contoh:
  python src/evaluate.py --split val --run-name baseline
  python src/evaluate.py --split test --confirm-final
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (FINAL_DIR, IDX_TO_CLASS, RESULTS_DIR, SELECTION_LOCK, SPLIT_DIR,
                    TEST_ACCESS_LOG)
from metrics import classification_metrics

THRESHOLD = 0.5  # tetap; bukan hyperparameter yang boleh dituning pada test


# --------------------------------------------------------------------------- #
# Kebijakan akses test set (tanpa torch)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_access_log(path: Path = TEST_ACCESS_LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def enforce_test_policy(run_name: str | None, allow_rerun: bool,
                        lock_path: Path = SELECTION_LOCK, log_path: Path = TEST_ACCESS_LOG,
                        runs_dir: Path = RESULTS_DIR / "runs") -> tuple[dict, Path]:
    """Hentikan program jika syarat evaluasi test tidak terpenuhi. Return (lock, checkpoint)."""
    if not lock_path.exists():
        sys.exit("[DITOLAK] Konfigurasi final belum dikunci. Jalankan dulu:\n"
                 "  python src/compare_runs.py --select <run> --reason \"...\"\n"
                 "Test set tidak boleh dibuka sebelum arsitektur & hyperparameter final ditentukan "
                 "lewat validation set.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    run = run_name or lock["selected_run"]
    if run != lock["selected_run"]:
        sys.exit(f"[DITOLAK] Run '{run}' bukan run yang dikunci ('{lock['selected_run']}'). "
                 "Test set hanya boleh dipakai untuk satu konfigurasi final.")
    ckpt = runs_dir / run / "best_model.pt"
    if not ckpt.exists():
        sys.exit(f"[DITOLAK] Checkpoint tidak ada: {ckpt}")
    if sha256_file(ckpt) != lock["checkpoint_sha256"]:
        sys.exit("[DITOLAK] SHA-256 checkpoint berbeda dari saat dikunci. Model berubah setelah "
                 "dipilih; ulangi pemilihan (compare_runs.py) tanpa melihat test.")
    previous = [e for e in read_access_log(log_path) if e.get("event") == "test_evaluation_started"]
    if previous and not allow_rerun:
        sys.exit(f"[DITOLAK] Test set sudah pernah dievaluasi pada {previous[0]['timestamp_utc']}. "
                 "Evaluasi akhir hanya boleh sekali. Gunakan --allow-rerun hanya jika benar-benar perlu "
                 "(akan tercatat sebagai rerun di log).")
    return lock, ckpt


def append_access_log(entry: dict, path: Path = TEST_ACCESS_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Inferensi (torch di-import di sini saja)
# --------------------------------------------------------------------------- #
def predict(ckpt_path: Path, split: str, batch_size: int, num_workers: int,
            device_pref: str, on_ready=None) -> dict:
    import torch
    from config import ModelConfig
    from model import SmallCNN
    from preprocessing import (load_norm_stats, make_loader, sanity_check_batch,
                               verify_split_integrity)
    from utils import device_description, get_device, set_seed

    verify_split_integrity(("train", "val", "test") if split == "test" else ("train", "val"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mc = dict(ckpt["model_config"])
    mc["conv_channels"] = tuple(mc["conv_channels"])
    model_cfg = ModelConfig(**mc)

    mean, std = load_norm_stats()
    if not (np.allclose(mean, ckpt["norm_mean"]) and np.allclose(std, ckpt["norm_std"])):
        sys.exit("[ERROR] Statistik normalisasi di checkpoint berbeda dari norm_stats.json.")

    set_seed(ckpt.get("seed", 42))
    device = get_device(device_pref)
    model = SmallCNN(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    loader = make_loader(split, batch_size, train=False, num_workers=num_workers,
                         allow_test=(split == "test"))
    paths = [p for p, _ in loader.dataset.samples]  # urutan = urutan CSV (shuffle=False)
    if on_ready:
        on_ready(len(paths))

    probs, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i == 0:
                sanity_check_batch(x, y, size=model_cfg.input_size, channels=model_cfg.in_channels)
            logits = model(x.to(device)).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return {
        "paths": paths, "y_true": np.concatenate(labels).astype(int), "y_prob": np.concatenate(probs),
        "device": device_description(device), "inference_seconds": round(time.perf_counter() - t0, 2),
        "checkpoint_epoch": ckpt.get("epoch"), "torch": torch.__version__,
    }


# --------------------------------------------------------------------------- #
# Laporan (tanpa torch)
# --------------------------------------------------------------------------- #
def write_predictions_csv(path: Path, paths, y_true, y_prob, threshold: float = THRESHOLD) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "prob_dog", "pred", "correct"])
        for p, y, pr in zip(paths, y_true, y_prob):
            pred = int(pr >= threshold)
            w.writerow([p, IDX_TO_CLASS[int(y)], f"{pr:.6f}", IDX_TO_CLASS[pred], int(pred == int(y))])


def plot_confusion(path: Path, cm: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    names = [IDX_TO_CLASS[0], IDX_TO_CLASS[1]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, fmt, sub in ((axes[0], mat, "{:d}", "Jumlah"), (axes[1], norm, "{:.1%}", "Dinormalisasi per baris (recall)")):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], [f"pred {n}" for n in names])
        ax.set_yticks([0, 1], [f"aktual {n}" for n in names])
        ax.set_title(sub)
        for i in range(2):
            for j in range(2):
                v = data[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center",
                        color="white" if v > data.max() / 2 else "black", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics_md(path: Path, m: dict, title: str, meta: dict) -> None:
    cm = m["confusion_matrix"]
    L = [f"# {title}\n",
         f"- Run: **{meta['run']}** (checkpoint epoch {meta['checkpoint_epoch']})",
         f"- Jumlah gambar: {m['n']} (Cat {m['class_counts']['Cat']}, Dog {m['class_counts']['Dog']})",
         f"- Threshold: {m['threshold']} (tetap, tidak dituning) | Kelas positif: {m['positive_class']}",
         f"- Device: {meta['device']} | waktu inferensi: {meta['inference_seconds']} detik",
         f"- Waktu evaluasi (UTC): {meta['timestamp_utc']}\n",
         "## Metrik ringkas\n",
         "| Metrik | Nilai |", "|---|---:|",
         f"| Accuracy | {m['accuracy']:.4f} |",
         f"| Precision (Dog) | {m['precision_dog']:.4f} |",
         f"| Recall (Dog) | {m['recall_dog']:.4f} |",
         f"| F1-score (Dog) | {m['f1_dog']:.4f} |",
         f"| Balanced accuracy | {m['balanced_accuracy']:.4f} |",
         f"| Macro F1 | {m['macro_f1']:.4f} |",
         f"| ROC-AUC | {m['roc_auc']:.4f} |",
         f"| BCE loss | {m['bce_loss']:.4f} |\n",
         "## Metrik per kelas\n",
         "| Kelas | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
         f"| Cat | {m['precision_cat']:.4f} | {m['recall_cat']:.4f} | {m['f1_cat']:.4f} | {m['class_counts']['Cat']} |",
         f"| Dog | {m['precision_dog']:.4f} | {m['recall_dog']:.4f} | {m['f1_dog']:.4f} | {m['class_counts']['Dog']} |\n",
         "## Confusion matrix (baris = aktual, kolom = prediksi)\n",
         "| | pred Cat | pred Dog |", "|---|---:|---:|",
         f"| **aktual Cat** | {cm['tn']} (TN) | {cm['fp']} (FP) |",
         f"| **aktual Dog** | {cm['fn']} (FN) | {cm['tp']} (TP) |\n",
         f"- FP = Cat yang diprediksi Dog: {cm['fp']}; FN = Dog yang diprediksi Cat: {cm['fn']}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluasi model pada validation atau test set.")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--run-name", default=None, help="Default: run terkunci (test) / 'baseline' (val)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--confirm-final", action="store_true",
                    help="WAJIB untuk --split test: konfirmasi bahwa ini evaluasi akhir")
    ap.add_argument("--allow-rerun", action="store_true", help="Izinkan evaluasi test ulang (dicatat sebagai rerun)")
    args = ap.parse_args()

    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731

    if args.split == "test":
        if not args.confirm_final:
            sys.exit("[DITOLAK] Evaluasi test set adalah evaluasi AKHIR dan hanya boleh sekali.\n"
                     "Pastikan konfigurasi final sudah dikunci, lalu jalankan dengan --confirm-final.")
        lock, ckpt = enforce_test_policy(args.run_name, args.allow_rerun)
        run, out_dir = lock["selected_run"], FINAL_DIR
        title = "Hasil Evaluasi Akhir pada TEST SET"

        def on_ready(n_images: int) -> None:  # dipanggil tepat sebelum gambar test dibaca
            prior = [e for e in read_access_log() if e.get("event") == "test_evaluation_started"]
            append_access_log({
                "event": "test_evaluation_started", "timestamp_utc": now(), "run": run,
                "checkpoint_sha256": lock["checkpoint_sha256"],
                "test_csv_sha256": sha256_file(SPLIT_DIR / "test.csv"),
                "n_images": n_images, "rerun": bool(prior), "host": platform.node(),
            })
    else:
        run = args.run_name or "baseline"
        ckpt = RESULTS_DIR / "runs" / run / "best_model.pt"
        if not ckpt.exists():
            sys.exit(f"[ERROR] Checkpoint tidak ada: {ckpt}")
        out_dir = RESULTS_DIR / "runs" / run
        title = "Hasil Evaluasi pada VALIDATION SET"
        on_ready = None

    out_dir.mkdir(parents=True, exist_ok=True)
    res = predict(ckpt, args.split, args.batch_size, args.num_workers, args.device, on_ready=on_ready)

    m = classification_metrics(res["y_true"], res["y_prob"], THRESHOLD)
    meta = {"run": run, "checkpoint_epoch": res["checkpoint_epoch"], "device": res["device"],
            "inference_seconds": res["inference_seconds"], "timestamp_utc": now()}
    (out_dir / f"{args.split}_metrics.json").write_text(
        json.dumps({"split": args.split, "meta": meta, "metrics": m}, indent=2), encoding="utf-8")
    write_metrics_md(out_dir / f"{args.split}_metrics.md", m, title, meta)
    write_predictions_csv(out_dir / f"{args.split}_predictions.csv", res["paths"], res["y_true"], res["y_prob"])
    plot_confusion(out_dir / f"confusion_matrix_{args.split}.png", m["confusion_matrix"],
                   f"Confusion matrix - {args.split} set ({run})")
    if args.split == "test":
        append_access_log({"event": "test_evaluation_finished", "timestamp_utc": now(), "run": run,
                           "accuracy": m["accuracy"], "f1_dog": m["f1_dog"]})

    cm = m["confusion_matrix"]
    print(f"=== {title} ===")
    print(f"Run {run} | n={m['n']} | device: {res['device']}")
    print(f"Accuracy {m['accuracy']:.4f} | Precision(Dog) {m['precision_dog']:.4f} | "
          f"Recall(Dog) {m['recall_dog']:.4f} | F1(Dog) {m['f1_dog']:.4f} | AUC {m['roc_auc']:.4f}")
    print(f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"Output: {out_dir.resolve()}")
    if args.split == "test":
        print("Test set sudah dievaluasi dan tercatat di log. Jangan ubah model/konfigurasi lagi "
              "berdasarkan hasil ini. Lanjut: python src/error_analysis.py")


if __name__ == "__main__":
    main()
