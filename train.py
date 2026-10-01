#!/usr/bin/env python3
"""
train.py
--------
Training CNN Cat vs Dog + monitoring pada validation set.

ATURAN ISOLASI TEST SET
  Script ini HANYA membaca train.csv dan val.csv. Test set tidak dibuka sama sekali.
  Pemilihan epoch terbaik, early stopping, dan perbandingan konfigurasi memakai
  validation loss/accuracy saja.

Yang disimpan di results/runs/<run-name>/:
  config.json           seluruh konfigurasi eksperimen (arsitektur, optimizer, LR, device, seed, ...)
  history.csv           epoch, train_loss, val_loss, train_acc, val_acc, lr, epoch_time_s
  training_curves.png   kurva loss & accuracy (train vs val)
  best_model.pt         checkpoint dengan val loss terendah
  last_model.pt         checkpoint epoch terakhir
  train_summary.md      ringkasan siap salin ke laporan

Catatan metrik: train_loss/train_acc dihitung selama training (running average) dengan
augmentasi dan dropout aktif, sehingga biasanya sedikit lebih buruk daripada nilai val.

Contoh:
  python src/train.py                                  # konfigurasi default (run 'baseline')
  python src/train.py --run-name lr3e-4 --lr 3e-4      # eksperimen lain (dibandingkan lewat val)
  python src/train.py --run-name no_aug --no-augment
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from config import DEFAULT_MODEL, RESULTS_DIR, SEED
from model import SmallCNN, count_parameters
from preprocessing import (load_norm_stats, make_loader, sanity_check_batch,
                           verify_split_integrity)
from utils import device_description, get_device, set_seed


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizer tidak dikenal: {name}")


# --------------------------------------------------------------------------- #
# Satu epoch (train jika optimizer diberikan, selain itu evaluasi)
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, check_first_batch=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if check_first_batch and i == 0:
                sanity_check_batch(x, y)  # shape/dtype/label dicek sebelum masuk model
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).squeeze(1)          # (N,1) -> (N,)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            correct += ((logits > 0).float() == y).sum().item()  # logit > 0  <=>  P(Dog) > 0.5
            n += bs
    return total_loss / n, correct / n


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_history_csv(path: Path, history: list[dict]) -> None:
    fields = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(history)


def plot_curves(path: Path, history: list[dict], best_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], marker="o", ms=3, label="train loss")
    axes[0].plot(ep, [h["val_loss"] for h in history], marker="o", ms=3, label="validation loss")
    axes[0].set_title("Loss vs epoch")
    axes[0].set_ylabel("BCE loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], marker="o", ms=3, label="train accuracy")
    axes[1].plot(ep, [h["val_acc"] for h in history], marker="o", ms=3, label="validation accuracy")
    axes[1].set_title("Accuracy vs epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"epoch terbaik ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, cfg: dict, best: dict, total_time: float, stopped_early: bool) -> None:
    L = ["# Ringkasan Training\n",
         "## Konfigurasi eksperimen\n",
         "| Item | Nilai |", "|---|---|",
         f"| Run name | {cfg['run_name']} |",
         f"| Arsitektur | {cfg['architecture']} |",
         f"| Input shape | {cfg['input_shape']} |",
         f"| Konfigurasi konvolusi | kernel {cfg['model']['kernel_size']}x{cfg['model']['kernel_size']}, "
         f"padding {cfg['model']['padding']}, stride {cfg['model']['stride']}, "
         f"filter {list(cfg['model']['conv_channels'])} |",
         "| Activation | ReLU (output: sigmoid pada logit tunggal) |",
         f"| Pooling | MaxPool {cfg['model']['pool_size']}x{cfg['model']['pool_size']} "
         f"(stride {cfg['model']['pool_stride']}) + GlobalAvgPool |",
         f"| Jumlah parameter (trainable) | {cfg['trainable_params']:,} |",
         f"| Optimizer | {cfg['optimizer']} (weight decay {cfg['weight_decay']}) |",
         f"| Loss | {cfg['loss']} |",
         f"| Learning rate | {cfg['learning_rate']} (konstan) |",
         f"| Device | {cfg['device']} |",
         f"| Batch size | {cfg['batch_size']} |",
         f"| Epoch maksimum | {cfg['epochs']} (dijalankan: {cfg['epochs_run']}"
         f"{', early stopping' if stopped_early else ''}) |",
         f"| Augmentasi train | {cfg['augment']} |",
         f"| Random seed | {cfg['seed']} |",
         f"| Waktu training total | {total_time:.1f} detik ({total_time / 60:.1f} menit) |",
         f"| Python / PyTorch | {cfg['python']} / {cfg['torch']} |",
         "", "## Hasil validasi (epoch terbaik berdasarkan val loss)\n",
         "| Epoch | Train loss | Val loss | Train acc | Val acc |", "|---:|---:|---:|---:|---:|",
         f"| {best['epoch']} | {best['train_loss']:.4f} | {best['val_loss']:.4f} | "
         f"{best['train_acc']:.4f} | {best['val_acc']:.4f} |",
         "", "Kurva: `training_curves.png`; riwayat per epoch: `history.csv`.",
         "Test set tidak diakses pada tahap ini.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Training CNN Cat vs Dog.")
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stopping: berhenti jika val loss tidak membaik selama N epoch (0 = nonaktif)")
    ap.add_argument("--no-augment", action="store_true", help="Matikan augmentasi pada training")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--overwrite", action="store_true", help="Timpa run dengan nama yang sama")
    args = ap.parse_args()

    run_dir = RESULTS_DIR / "runs" / args.run_name
    if (run_dir / "history.csv").exists() and not args.overwrite:
        sys.exit(f"[ERROR] Run '{args.run_name}' sudah ada di {run_dir}. "
                 "Gunakan --run-name lain atau --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # hanya train & val yang diverifikasi/dibuka; test.csv tidak disentuh
    verify_split_integrity(("train", "val"))
    set_seed(args.seed)
    device = get_device(args.device)
    mean, std = load_norm_stats()

    train_loader = make_loader("train", args.batch_size, train=True, num_workers=args.num_workers,
                               augment=not args.no_augment, seed=args.seed)
    val_loader = make_loader("val", args.batch_size, train=False, num_workers=args.num_workers)

    model = SmallCNN(DEFAULT_MODEL).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    cfg = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": "SmallCNN: 4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2] -> GAP -> Dropout -> Linear(1)",
        "input_shape": f"{DEFAULT_MODEL.in_channels}x{DEFAULT_MODEL.input_size}x{DEFAULT_MODEL.input_size} (C x H x W)",
        "model": asdict(DEFAULT_MODEL),
        "trainable_params": count_parameters(model),
        "optimizer": args.optimizer, "weight_decay": args.weight_decay,
        "loss": "BCEWithLogitsLoss", "learning_rate": args.lr,
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "augment": not args.no_augment, "seed": args.seed,
        "device": device_description(device), "num_workers": args.num_workers,
        "normalization_mean": mean, "normalization_std": std,
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "python": platform.python_version(), "torch": torch.__version__,
        "test_set_accessed": False,
    }
    print(f"[INFO] Run        : {args.run_name}")
    print(f"[INFO] Device     : {cfg['device']}")
    print(f"[INFO] Parameter  : {cfg['trainable_params']:,}")
    print(f"[INFO] Data       : train={cfg['n_train']}  val={cfg['n_val']}  (test tidak dibuka)")

    history, best = [], None
    best_val_loss, bad_epochs, stopped_early = float("inf"), 0, False
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    check_first_batch=(epoch == 1))
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                    check_first_batch=(epoch == 1))
        dt = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc,
               "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(dt, 2)}
        history.append(row)

        improved = va_loss < best_val_loss
        if improved:
            best_val_loss, best, bad_epochs = va_loss, row, 0
            torch.save({"model_state": model.state_dict(), "model_config": asdict(DEFAULT_MODEL),
                        "epoch": epoch, "val_loss": va_loss, "val_acc": va_acc,
                        "norm_mean": mean, "norm_std": std, "seed": args.seed},
                       run_dir / "best_model.pt")
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:>3}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s{'  *best*' if improved else ''}",
              flush=True)
        save_history_csv(run_dir / "history.csv", history)  # simpan tiap epoch (aman jika terhenti)

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[INFO] Early stopping: val loss tidak membaik selama {args.patience} epoch.")
            stopped_early = True
            break

    total_time = time.perf_counter() - t_start
    torch.save({"model_state": model.state_dict(), "model_config": asdict(DEFAULT_MODEL),
                "epoch": history[-1]["epoch"], "norm_mean": mean, "norm_std": std, "seed": args.seed},
               run_dir / "last_model.pt")

    cfg.update({"epochs_run": len(history), "stopped_early": stopped_early,
                "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                "best_val_acc": best["val_acc"], "training_time_seconds": round(total_time, 1)})
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    plot_curves(run_dir / "training_curves.png", history, best["epoch"])
    write_summary(run_dir / "train_summary.md", cfg, best, total_time, stopped_early)

    print(f"\n=== SELESAI ({total_time / 60:.1f} menit) ===")
    print(f"Epoch terbaik: {best['epoch']} | val loss {best['val_loss']:.4f} | val acc {best['val_acc']:.4f}")
    print(f"Output: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
