#!/usr/bin/env python3
"""
manual_calc.py
--------------
Perhitungan MANUAL output shape dan jumlah parameter setiap layer CNN, lalu
dibandingkan dengan model summary dari PyTorch (jika torch terpasang).

Rumus (sesuai task):
  Output conv/pool : floor((N + 2P - K) / S) + 1
  Parameter conv   : (K x K x C_in x C_out) + C_out
  Parameter BN     : 2 x C  (gamma & beta, trainable; running mean/var adalah buffer)
  Parameter Linear : (in x out) + out
  Pooling / ReLU / GAP / Dropout : 0 parameter

Bagian perhitungan murni Python (tanpa torch). Perbandingan ke framework dijalankan
otomatis jika torch tersedia, atau paksa dengan --compare / matikan dengan --no-compare.

Output (results/model/):
  manual_calc.md              tabel manual (siap salin ke laporan)
  manual_vs_framework.md/json hasil perbandingan manual vs model summary
  architecture.md             diagram teks arsitektur

Contoh:
  python src/manual_calc.py
  python src/manual_calc.py --no-compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import DEFAULT_MODEL, ModelConfig


# --------------------------------------------------------------------------- #
# Rumus
# --------------------------------------------------------------------------- #
def out_size(n: int, k: int, p: int, s: int) -> int:
    """floor((N + 2P - K) / S) + 1"""
    return (n + 2 * p - k) // s + 1


def conv_params(k: int, c_in: int, c_out: int) -> int:
    """(K x K x C_in x C_out) + C_out"""
    return k * k * c_in * c_out + c_out


def bn_params(c: int) -> int:
    return 2 * c


def linear_params(n_in: int, n_out: int) -> int:
    return n_in * n_out + n_out


# --------------------------------------------------------------------------- #
# Tabel manual untuk seluruh model
# --------------------------------------------------------------------------- #
def manual_table(cfg: ModelConfig) -> list[dict]:
    """Daftar layer: name, type, out_shape (H, W, C), params, formula (string penjelasan).
    Nama layer HARUS sama dengan nama modul di model.py agar bisa dibandingkan."""
    rows = []
    h = w = cfg.input_size
    c = cfg.in_channels
    rows.append({"name": "input", "type": "Input", "out_shape": (h, w, c), "params": 0,
                 "formula": f"{h} x {w} x {c}"})

    for i, c_out in enumerate(cfg.conv_channels, start=1):
        k, p, s = cfg.kernel_size, cfg.padding, cfg.stride
        h_new = out_size(h, k, p, s)
        w_new = out_size(w, k, p, s)
        n_par = conv_params(k, c, c_out)
        rows.append({
            "name": f"conv{i}", "type": "Conv2d", "out_shape": (h_new, w_new, c_out), "params": n_par,
            "formula": (f"out = floor(({h}+2*{p}-{k})/{s})+1 = {h_new}; "
                        f"params = ({k}*{k}*{c}*{c_out})+{c_out} = {n_par}"),
        })
        h, w, c = h_new, w_new, c_out

        rows.append({"name": f"bn{i}", "type": "BatchNorm2d", "out_shape": (h, w, c), "params": bn_params(c),
                     "formula": f"2*{c} = {bn_params(c)} (gamma, beta)"})
        rows.append({"name": f"relu{i}", "type": "ReLU", "out_shape": (h, w, c), "params": 0,
                     "formula": "elementwise, shape tetap"})

        pk, ps = cfg.pool_size, cfg.pool_stride
        h_new = out_size(h, pk, 0, ps)
        w_new = out_size(w, pk, 0, ps)
        rows.append({"name": f"pool{i}", "type": "MaxPool2d", "out_shape": (h_new, w_new, c), "params": 0,
                     "formula": f"out = floor(({h}+0-{pk})/{ps})+1 = {h_new}"})
        h, w = h_new, w_new

    rows.append({"name": "gap", "type": "AdaptiveAvgPool2d", "out_shape": (1, 1, c), "params": 0,
                 "formula": f"rata-rata spasial {h}x{w} -> 1x1"})
    rows.append({"name": "flatten", "type": "Flatten", "out_shape": (1, 1, c), "params": 0,
                 "formula": f"1x1x{c} -> vektor {c}"})
    rows.append({"name": "dropout", "type": "Dropout", "out_shape": (1, 1, c), "params": 0,
                 "formula": f"p={cfg.dropout}, tidak ada parameter"})
    n_fc = linear_params(c, 1)
    rows.append({"name": "fc", "type": "Linear", "out_shape": (1, 1, 1), "params": n_fc,
                 "formula": f"({c}*1)+1 = {n_fc} -> 1 logit; P(Dog)=sigmoid(logit)"})
    return rows


def fmt_shape(shape) -> str:
    h, w, c = shape
    return f"{h} x {w} x {c}"


# --------------------------------------------------------------------------- #
# Contoh wajib dari task: 32x32x3, 16 filter, K=3, P=1, S=1
# --------------------------------------------------------------------------- #
def task_example() -> dict:
    n, c_in, c_out, k, p, s = 32, 3, 16, 3, 1, 1
    return {
        "input": (n, n, c_in), "filters": c_out, "kernel": k, "padding": p, "stride": s,
        "output_height": out_size(n, k, p, s), "output_width": out_size(n, k, p, s),
        "output_channels": c_out, "params": conv_params(k, c_in, c_out),
        "weights_only": k * k * c_in * c_out, "biases": c_out,
    }


# --------------------------------------------------------------------------- #
# Perbandingan manual vs framework (fungsi murni, bisa diuji tanpa torch)
# --------------------------------------------------------------------------- #
def compare(manual: list[dict], framework: dict[str, dict]) -> list[dict]:
    """framework: {nama_layer: {"out_shape": (H, W, C), "params": int}}"""
    result = []
    for row in manual:
        if row["name"] == "input":
            continue
        fw = framework.get(row["name"])
        if fw is None:
            result.append({**row, "fw_shape": None, "fw_params": None, "match": False,
                           "note": "layer tidak ditemukan di model"})
            continue
        ok = tuple(fw["out_shape"]) == tuple(row["out_shape"]) and int(fw["params"]) == int(row["params"])
        result.append({**row, "fw_shape": tuple(fw["out_shape"]), "fw_params": int(fw["params"]),
                       "match": ok, "note": ""})
    return result


def framework_from_torch(cfg: ModelConfig) -> dict[str, dict]:
    from model import SmallCNN, layer_report  # impor lazy: butuh torch
    model = SmallCNN(cfg)
    return layer_report(model, (cfg.in_channels, cfg.input_size, cfg.input_size))


def task_example_from_torch() -> dict:
    import torch
    from torch import nn
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    y = conv(torch.zeros(1, 3, 32, 32))
    return {"output_shape_nchw": tuple(y.shape), "params": sum(p.numel() for p in conv.parameters())}


# --------------------------------------------------------------------------- #
# Penulisan laporan
# --------------------------------------------------------------------------- #
def write_manual_md(path: Path, cfg: ModelConfig, rows: list[dict], ex: dict) -> None:
    total = sum(r["params"] for r in rows)
    conv_total = sum(r["params"] for r in rows if r["type"] == "Conv2d")
    L = ["# Perhitungan Manual Output Shape & Parameter\n",
         "Rumus: `Output = floor((N + 2P - K) / S) + 1`, `Param conv = (K x K x C_in x C_out) + C_out`\n",
         "## A. Contoh wajib task (32 x 32 x 3)\n",
         f"- Input: {fmt_shape(ex['input'])}; {ex['filters']} filter, kernel {ex['kernel']}x{ex['kernel']}, "
         f"padding {ex['padding']}, stride {ex['stride']}",
         f"- Output height = floor(({ex['input'][0]} + 2*{ex['padding']} - {ex['kernel']}) / {ex['stride']}) + 1 "
         f"= **{ex['output_height']}**",
         f"- Output width  = **{ex['output_width']}**",
         f"- Output channels = jumlah filter = **{ex['output_channels']}**",
         f"- Parameter = ({ex['kernel']}x{ex['kernel']}x{ex['input'][2]}x{ex['filters']}) + {ex['filters']} "
         f"= {ex['weights_only']} + {ex['biases']} = **{ex['params']}**",
         f"- Output shape: **{ex['output_height']} x {ex['output_width']} x {ex['output_channels']}**\n",
         f"## B. Model utama (input {cfg.input_size} x {cfg.input_size} x {cfg.in_channels})\n",
         "| Layer | Tipe | Output shape (H x W x C) | Parameter | Perhitungan |",
         "|---|---|---|---:|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['type']} | {fmt_shape(r['out_shape'])} | {r['params']:,} | {r['formula']} |")
    L += ["", f"- Total parameter conv: **{conv_total:,}**",
          f"- Total parameter model (trainable): **{total:,}**", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_compare_md(path: Path, cmp_rows: list[dict], ex: dict, ex_fw: dict | None) -> None:
    all_ok = all(r["match"] for r in cmp_rows)
    L = ["# Perbandingan Manual vs Model Summary (PyTorch)\n",
         f"Status keseluruhan: **{'SEMUA COCOK' if all_ok else 'ADA PERBEDAAN'}**\n",
         "| Layer | Shape manual | Shape framework | Param manual | Param framework | Cocok |",
         "|---|---|---|---:|---:|:---:|"]
    for r in cmp_rows:
        fws = fmt_shape(r["fw_shape"]) if r["fw_shape"] else "-"
        fwp = f"{r['fw_params']:,}" if r["fw_params"] is not None else "-"
        L.append(f"| {r['name']} | {fmt_shape(r['out_shape'])} | {fws} | {r['params']:,} | {fwp} "
                 f"| {'ya' if r['match'] else 'TIDAK'} |")
    L += ["", f"Total parameter manual: **{sum(r['params'] for r in cmp_rows):,}** | "
              f"framework: **{sum(r['fw_params'] or 0 for r in cmp_rows):,}**\n"]
    if ex_fw:
        n, h, w = ex_fw["output_shape_nchw"][1:]
        L += ["## Contoh task 32x32x3 -> 16 filter",
              f"- Manual : {ex['output_height']} x {ex['output_width']} x {ex['output_channels']}, "
              f"{ex['params']} parameter",
              f"- PyTorch: {w} x {h} x {n} (HxWxC), {ex_fw['params']} parameter",
              f"- Cocok: **{'ya' if (h, w, n, ex_fw['params']) == (ex['output_height'], ex['output_width'], ex['output_channels'], ex['params']) else 'TIDAK'}**", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_architecture_md(path: Path, cfg: ModelConfig, rows: list[dict]) -> None:
    L = ["# Arsitektur CNN\n", "```", f"Input  {cfg.input_size}x{cfg.input_size}x{cfg.in_channels}"]
    for i, c_out in enumerate(cfg.conv_channels, start=1):
        conv = next(r for r in rows if r["name"] == f"conv{i}")
        pool = next(r for r in rows if r["name"] == f"pool{i}")
        L += ["  |", f"  v  Block {i}: Conv2d({cfg.kernel_size}x{cfg.kernel_size}, {c_out} filter, "
                      f"pad={cfg.padding}, stride={cfg.stride}) -> BatchNorm -> ReLU -> MaxPool({cfg.pool_size}x{cfg.pool_size})",
              f"     conv out {fmt_shape(conv['out_shape'])} -> pool out {fmt_shape(pool['out_shape'])}"]
    L += ["  |", f"  v  GlobalAvgPool -> Flatten ({cfg.conv_channels[-1]})", "  |",
          f"  v  Dropout(p={cfg.dropout})", "  |", "  v  Linear(%d -> 1)  = 1 logit" % cfg.conv_channels[-1],
          "  |", "  v  sigmoid -> P(Dog)   (Cat=0, Dog=1)", "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Hitung manual output shape & parameter CNN.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--compare", action="store_true", help="Wajib bandingkan dengan torch (error jika tidak ada)")
    g.add_argument("--no-compare", action="store_true", help="Lewati perbandingan dengan torch")
    ap.add_argument("--out-dir", type=Path, default=Path("results/model"))
    args = ap.parse_args()

    cfg = DEFAULT_MODEL
    rows = manual_table(cfg)
    ex = task_example()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_manual_md(args.out_dir / "manual_calc.md", cfg, rows, ex)
    write_architecture_md(args.out_dir / "architecture.md", cfg, rows)

    print("=== CONTOH TASK: 32x32x3 -> 16 filter, K=3, P=1, S=1 ===")
    print(f"Output : {ex['output_height']} x {ex['output_width']} x {ex['output_channels']}")
    print(f"Param  : {ex['params']}  (= {ex['weights_only']} bobot + {ex['biases']} bias)")
    print(f"\n=== MODEL UTAMA (input {cfg.input_size}x{cfg.input_size}x{cfg.in_channels}) ===")
    for r in rows:
        print(f"{r['name']:<8}{r['type']:<20}{fmt_shape(r['out_shape']):<16}{r['params']:>10,}")
    print(f"Total parameter (manual): {sum(r['params'] for r in rows):,}")

    if args.no_compare:
        print("\n[INFO] Perbandingan dengan framework dilewati (--no-compare).")
        return
    try:
        framework = framework_from_torch(cfg)
        ex_fw = task_example_from_torch()
    except ImportError as e:
        if args.compare:
            sys.exit(f"[ERROR] torch dibutuhkan untuk --compare: {e}")
        print(f"\n[PERINGATAN] torch tidak tersedia ({e}); perbandingan dengan model summary dilewati.\n"
              "             Jalankan ulang di environment dengan torch terpasang.")
        return

    cmp_rows = compare(rows, framework)
    write_compare_md(args.out_dir / "manual_vs_framework.md", cmp_rows, ex, ex_fw)
    (args.out_dir / "manual_vs_framework.json").write_text(
        json.dumps({"layers": cmp_rows, "task_example_manual": ex, "task_example_torch": ex_fw},
                   indent=2, default=list), encoding="utf-8")
    bad = [r["name"] for r in cmp_rows if not r["match"]]
    print("\n=== PERBANDINGAN MANUAL vs MODEL SUMMARY ===")
    print("SEMUA LAYER COCOK" if not bad else f"PERBEDAAN pada layer: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
