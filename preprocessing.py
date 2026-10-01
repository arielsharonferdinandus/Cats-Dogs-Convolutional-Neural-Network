#!/usr/bin/env python3
"""
preprocessing.py
----------------
Pipeline preprocessing yang konsisten untuk train / validation / test.

Keputusan desain (didokumentasikan juga di results/preprocessing/preprocessing_summary.md):
  * Target ukuran      : 128 x 128 (resize langsung, bilinear + antialias, aspect ratio tidak dijaga)
  * Channel            : 3 (semua gambar dikonversi ke RGB -> menangani grayscale/RGBA/palette)
  * Normalisasi        : skala ke [0, 1] lalu standardisasi per-channel (x - mean) / std.
                         mean/std dihitung HANYA dari training set (tidak ada kebocoran val/test).
  * Label              : Cat = 0, Dog = 1 (float32, 1 logit; sigmoid(logit) = P(Dog))
  * Train              : Resize -> [augmentasi: HorizontalFlip, Rotation 10 derajat] -> ToTensor -> Normalize
  * Validation / Test  : Resize -> ToTensor -> Normalize   (TANPA augmentasi)

Isolasi test set: read_split_csv("test") dan CatsDogsDataset("test") melempar
PermissionError kecuali allow_test=True. Hanya evaluate.py / error_analysis.py yang boleh
memakainya. verify_split_integrity() memeriksa SHA-256 CSV terhadap split_manifest.json.

Menjalankan modul ini (`python src/preprocessing.py`):
  1. memverifikasi integritas split,
  2. menghitung mean/std dari train -> data/splits/norm_stats.json,
  3. cek sanity satu batch (shape, dtype, nilai label),
  4. menyimpan contoh gambar hasil preprocessing & ringkasan Markdown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config import (CLASS_TO_IDX, DEFAULT_MODEL, IDX_TO_CLASS, NORM_STATS_PATH,
                    RESULTS_DIR, SPLIT_DIR, SPLIT_MANIFEST)

VALID_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Akses split + penjaga test set
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_split_integrity(splits: tuple[str, ...] = VALID_SPLITS) -> None:
    """Pastikan CSV split tidak berubah sejak split_data.py dijalankan."""
    if not SPLIT_MANIFEST.exists():
        sys.exit(f"[ERROR] {SPLIT_MANIFEST} tidak ada. Jalankan dulu: python src/split_data.py")
    expected = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))["split_csv_sha256"]
    for s in splits:
        actual = _sha256(SPLIT_DIR / f"{s}.csv")
        if actual != expected[s]:
            sys.exit(f"[ERROR] {s}.csv berubah sejak split dibuat (SHA-256 tidak cocok). "
                     "Jalankan ulang split_data.py dan seluruh eksperimen.")


def read_split_csv(split: str, allow_test: bool = False) -> list[dict]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split harus salah satu dari {VALID_SPLITS}, bukan '{split}'")
    if split == "test" and not allow_test:
        raise PermissionError(
            "Test set hanya boleh dibuka untuk evaluasi akhir (evaluate.py / error_analysis.py) "
            "dengan allow_test=True. Jangan dipakai untuk tuning.")
    with (SPLIT_DIR / f"{split}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def _resize(size: int) -> T.Resize:
    return T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)


def load_norm_stats(path: Path = NORM_STATS_PATH) -> tuple[list[float], list[float]]:
    if not path.exists():
        sys.exit(f"[ERROR] {path} tidak ada. Jalankan dulu: python src/preprocessing.py")
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["mean"], d["std"]


def build_transform(train: bool, size: int = DEFAULT_MODEL.input_size,
                    mean=None, std=None, augment: bool = True) -> T.Compose:
    """train=True -> boleh ada augmentasi. Val/test selalu deterministik."""
    if mean is None or std is None:
        mean, std = load_norm_stats()
    ops = [_resize(size)]
    if train and augment:
        ops += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=10)]
    ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CatsDogsDataset(Dataset):
    """Mengembalikan (tensor_gambar [3,H,W] float32, label float32 skalar)."""

    def __init__(self, split: str, transform, allow_test: bool = False):
        rows = read_split_csv(split, allow_test=allow_test)
        self.split = split
        self.transform = transform
        self.samples = [(r["path"], CLASS_TO_IDX[r["label"]]) for r in rows]  # KeyError jika label aneh

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")  # paksa 3 channel
        except Exception as e:  # noqa: BLE001
            # Jangan diam-diam melewati: file di split seharusnya sudah tervalidasi.
            raise RuntimeError(f"Gagal membaca gambar pada split '{self.split}': {path} ({e})") from e
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def make_loader(split: str, batch_size: int, train: bool, num_workers: int = 2,
                augment: bool = True, allow_test: bool = False, seed: int | None = None) -> DataLoader:
    ds = CatsDogsDataset(split, build_transform(train=train, augment=augment), allow_test=allow_test)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)


# --------------------------------------------------------------------------- #
# Verifikasi pipeline (dipakai juga untuk Bagian 10 - Data-Pipeline Verification)
# --------------------------------------------------------------------------- #
def sanity_check_batch(x: torch.Tensor, y: torch.Tensor, size: int = DEFAULT_MODEL.input_size,
                       channels: int = DEFAULT_MODEL.in_channels) -> None:
    """Assert shape/dtype/nilai. Gagal keras agar bug pipeline tidak lolos diam-diam."""
    assert x.ndim == 4 and tuple(x.shape[1:]) == (channels, size, size), \
        f"Shape gambar salah: {tuple(x.shape)}, diharapkan (N,{channels},{size},{size})"
    assert x.dtype == torch.float32, f"dtype gambar {x.dtype}, diharapkan float32"
    assert torch.isfinite(x).all(), "Ada NaN/Inf pada tensor gambar"
    assert y.ndim == 1 and y.shape[0] == x.shape[0], f"Shape label salah: {tuple(y.shape)}"
    assert set(y.unique().tolist()) <= {0.0, 1.0}, f"Nilai label tidak valid: {y.unique().tolist()}"


# --------------------------------------------------------------------------- #
# Statistik normalisasi (HANYA dari train)
# --------------------------------------------------------------------------- #
def compute_norm_stats(size: int) -> dict:
    rows = read_split_csv("train")
    resize = _resize(size)
    to_t = T.ToTensor()
    s = torch.zeros(3, dtype=torch.float64)
    sq = torch.zeros(3, dtype=torch.float64)
    n_pix = 0
    for i, r in enumerate(rows, 1):
        with Image.open(r["path"]) as im:
            t = to_t(resize(im.convert("RGB"))).double()  # [3,H,W] dalam [0,1]
        s += t.sum(dim=(1, 2))
        sq += (t ** 2).sum(dim=(1, 2))
        n_pix += t.shape[1] * t.shape[2]
        if i % 3000 == 0 or i == len(rows):
            print(f"  statistik normalisasi: {i}/{len(rows)}", flush=True)
    mean = s / n_pix
    std = (sq / n_pix - mean ** 2).clamp(min=0).sqrt()
    return {"mean": [round(float(v), 6) for v in mean], "std": [round(float(v), 6) for v in std],
            "computed_from": "train.csv saja", "n_images": len(rows), "image_size": size,
            "pixel_range_before_standardize": [0.0, 1.0]}


def save_sample_grid(out_path: Path, mean, std, n: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_tr = CatsDogsDataset("train", build_transform(train=True, mean=mean, std=std))
    ds_va = CatsDogsDataset("val", build_transform(train=False, mean=mean, std=std))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds_tr), size=n, replace=False)
    m = torch.tensor(mean).view(3, 1, 1)
    sd = torch.tensor(std).view(3, 1, 1)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for j, i in enumerate(idx):
        for row, ds, name in ((0, ds_tr, "train (augmentasi)"), (1, ds_va, "val (tanpa augmentasi)")):
            x, y = ds[int(i)]
            img = (x * sd + m).clamp(0, 1).permute(1, 2, 0).numpy()  # denormalisasi untuk tampilan
            axes[row, j].imshow(img)
            axes[row, j].set_title(f"{IDX_TO_CLASS[int(y.item())]} ({int(y.item())})", fontsize=9)
            axes[row, j].axis("off")
            if j == 0:
                axes[row, j].text(-0.05, 0.5, name, transform=axes[row, j].transAxes, rotation=90,
                                  ha="right", va="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_summary(path: Path, stats: dict, size: int) -> None:
    mean, std = stats["mean"], stats["std"]
    L = ["# Ringkasan Preprocessing\n",
         f"- Target image size: **{size} x {size}** (resize langsung, bilinear + antialias, aspect ratio tidak dijaga)",
         "- Jumlah channel: **3** (semua gambar dikonversi ke RGB)",
         f"- Normalisasi: skala piksel ke [0, 1], lalu `(x - mean) / std` per channel",
         f"  - mean (R,G,B) = {mean}",
         f"  - std  (R,G,B) = {std}",
         f"  - dihitung dari **training set saja** ({stats['n_images']} gambar) - tidak ada informasi val/test",
         "- Encoding label: **Cat = 0, Dog = 1**, float32; model memiliki 1 logit, `sigmoid(logit) = P(Dog)`;",
         "  loss `BCEWithLogitsLoss`",
         "",
         "| Split | Pipeline |",
         "|---|---|",
         f"| Train | Resize({size}) -> RandomHorizontalFlip(0.5) -> RandomRotation(10 derajat) -> ToTensor -> Normalize |",
         f"| Validation | Resize({size}) -> ToTensor -> Normalize |",
         f"| Test | Resize({size}) -> ToTensor -> Normalize (identik dengan validation) |",
         "",
         "Augmentasi hanya pada training. Statistik normalisasi yang sama dipakai untuk ketiga split.",
         "Contoh hasil: `sample_batch.png`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hitung statistik normalisasi & verifikasi pipeline.")
    ap.add_argument("--size", type=int, default=DEFAULT_MODEL.input_size)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "preprocessing")
    args = ap.parse_args()

    verify_split_integrity()
    print("[OK] Integritas split terverifikasi (SHA-256 cocok dengan manifest).")

    stats = compute_norm_stats(args.size)
    NORM_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[OK] Statistik normalisasi disimpan: {NORM_STATS_PATH}")
    print(f"     mean={stats['mean']}  std={stats['std']}")

    # cek sanity satu batch dari train & val (test TIDAK disentuh)
    for split, train in (("train", True), ("val", False)):
        loader = make_loader(split, batch_size=16, train=train, num_workers=0, seed=0)
        x, y = next(iter(loader))
        sanity_check_batch(x, y, size=args.size)
        print(f"[OK] Batch {split}: x{tuple(x.shape)} {x.dtype}, y{tuple(y.shape)} {y.dtype}, "
              f"label unik={sorted(y.unique().tolist())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_sample_grid(args.out_dir / "sample_batch.png", stats["mean"], stats["std"])
    write_summary(args.out_dir / "preprocessing_summary.md", stats, args.size)
    print(f"[OK] Ringkasan & contoh gambar: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
