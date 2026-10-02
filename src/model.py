#!/usr/bin/env python3
"""
model.py
--------
CNN kecil untuk klasifikasi biner Cat vs Dog (dilatih dari awal, tanpa transfer learning).

Arsitektur (konfigurasi di config.ModelConfig):
  Input 3x128x128
  [Conv2d(3x3, pad=1, stride=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)] x 4 blok
      filter: 32 -> 64 -> 128 -> 256
  AdaptiveAvgPool2d(1) (global average pooling) -> Flatten -> Dropout -> Linear(256 -> 1)
  Output: 1 logit; P(Dog) = sigmoid(logit). Loss: BCEWithLogitsLoss.

Nama modul (conv1, bn1, relu1, pool1, ..., gap, flatten, dropout, fc) sengaja sama
dengan nama pada manual_calc.py agar perbandingan manual vs framework bisa otomatis.

Jalankan `python src/model.py` untuk mencetak model summary.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from config import DEFAULT_MODEL, ModelConfig


class SmallCNN(nn.Module):
    def __init__(self, cfg: ModelConfig = DEFAULT_MODEL):
        super().__init__()
        self.cfg = cfg
        layers = OrderedDict()
        c_in = cfg.in_channels
        for i, c_out in enumerate(cfg.conv_channels, start=1):
            layers[f"conv{i}"] = nn.Conv2d(c_in, c_out, kernel_size=cfg.kernel_size,
                                           stride=cfg.stride, padding=cfg.padding)
            layers[f"bn{i}"] = nn.BatchNorm2d(c_out)
            layers[f"relu{i}"] = nn.ReLU(inplace=True)
            layers[f"pool{i}"] = nn.MaxPool2d(kernel_size=cfg.pool_size, stride=cfg.pool_stride)
            c_in = c_out
        layers["gap"] = nn.AdaptiveAvgPool2d(1)
        layers["flatten"] = nn.Flatten()
        layers["dropout"] = nn.Dropout(cfg.dropout)
        layers["fc"] = nn.Linear(c_in, 1)
        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> logit (N, 1). Sigmoid TIDAK diterapkan di sini."""
        return self.net(x)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def layer_report(model: nn.Module, input_chw: tuple[int, int, int]) -> dict[str, dict]:
    """Jalankan satu forward pass dan kumpulkan output shape (H, W, C) serta jumlah
    parameter tiap layer. Return: {nama_layer: {"out_shape": (H, W, C), "params": int}}"""
    report: dict[str, dict] = {}
    hooks = []

    def make_hook(name: str, module: nn.Module):
        def hook(_m, _inp, out):
            shape = tuple(out.shape[1:])  # buang dimensi batch
            if len(shape) == 3:           # (C, H, W) -> (H, W, C)
                c, h, w = shape
                hwc = (h, w, c)
            else:                         # (C,) setelah Flatten/Linear -> (1, 1, C)
                hwc = (1, 1, shape[0])
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            report[name] = {"out_shape": hwc, "params": n_params}
        return hook

    for name, module in model.net.named_children():
        hooks.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()  # BatchNorm/Dropout tidak mengubah shape; eval agar tidak memperbarui statistik
    device = next(model.parameters()).device
    model(torch.zeros(1, *input_chw, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return report


def print_summary(model: nn.Module, input_chw: tuple[int, int, int]) -> None:
    report = layer_report(model, input_chw)
    print(f"{'Layer':<10}{'Tipe':<20}{'Output (HxWxC)':<18}{'Param':>10}")
    print("-" * 58)
    for name, module in model.net.named_children():
        h, w, c = report[name]["out_shape"]
        print(f"{name:<10}{type(module).__name__:<20}{f'{h}x{w}x{c}':<18}{report[name]['params']:>10,}")
    print("-" * 58)
    print(f"Total parameter (trainable): {count_parameters(model):,}")


if __name__ == "__main__":
    cfg = DEFAULT_MODEL
    m = SmallCNN(cfg)
    print_summary(m, (cfg.in_channels, cfg.input_size, cfg.input_size))
