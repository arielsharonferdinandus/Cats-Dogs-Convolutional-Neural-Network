#!/usr/bin/env python3
"""
error_analysis.py
-----------------
Analisis kesalahan klasifikasi (misclassification) pada hasil evaluasi akhir test set.

Dijalankan SETELAH `evaluate.py --split test --confirm-final`. Input: results/final/test_predictions.csv.
Script ini tidak memuat model dan tidak mengubah apa pun pada model/konfigurasi.

Langkah:
  1. Ambil semua gambar yang salah diklasifikasikan; pisahkan Cat->Dog (FP) dan Dog->Cat (FN).
  2. Pilih contoh paling informatif: kesalahan dengan confidence tertinggi pada tiap jenis
     (--n-per-type, default 4 -> sampai 8 contoh; minimal 3 jika tersedia).
  3. Hitung petunjuk teknis objektif tiap gambar pada resolusi model (128x128):
     kecerahan, kontras, ketajaman (varians Laplacian), sisi terpendek gambar asli, aspect ratio.
  4. Bandingkan dengan sampel gambar yang BENAR (p10/p90) -> bendera: gelap, terang berlebih,
     kontras rendah, kabur, resolusi kecil, aspect ratio ekstrem. Bendera dihitung juga untuk
     SEMUA kesalahan untuk melihat faktor mana yang berlebihan (enriched) di antara error.
  5. Tulis laporan Markdown + grid gambar + CSV.

PENTING: petunjuk otomatis hanyalah HIPOTESIS. Faktor semantik (occlusion, pose tidak umum,
background membingungkan, kemiripan Cat/Dog, label noise) harus dikonfirmasi dengan melihat
gambar. Eksperimen lanjutan yang diusulkan harus dievaluasi pada VALIDATION set; hasil test
yang sudah dilihat tidak boleh dipakai untuk memilih model baru.

Output (results/final/):
  misclassification_analysis.md, misclassified_grid.png, misclassified_examples.csv,
  errors_all.csv, misclassified/*.jpg
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from config import FINAL_DIR

MODEL_SIDE = 128  # cue dihitung pada resolusi input model

FLAG_HINTS = {
    "gelap": ("pencahayaan rendah",
              "augmentasi brightness/contrast (ColorJitter) atau normalisasi histogram (CLAHE)"),
    "terang berlebih": ("overexposure / detail hilang pada area terang",
                        "augmentasi brightness-contrast yang lebih kuat"),
    "kontras rendah": ("kontras rendah / gambar berkabut sehingga tepi objek lemah",
                       "augmentasi contrast atau normalisasi histogram"),
    "kabur": ("blur / detail halus sedikit pada resolusi model",
              "augmentasi GaussianBlur & motion blur, atau naikkan resolusi input (mis. 160/224)"),
    "resolusi kecil": ("gambar asli kecil sehingga upsampling ke 128 px kehilangan detail",
                       "naikkan resolusi input atau filter/analisis gambar beresolusi sangat rendah"),
    "aspect ratio ekstrem": ("resize tanpa menjaga aspect ratio mendistorsi bentuk/crop kurang baik",
                             "resize dengan letterbox (padding) atau center-crop sebelum resize"),
}


# --------------------------------------------------------------------------- #
# Petunjuk teknis gambar
# --------------------------------------------------------------------------- #
def image_cues(path: str) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        small = np.asarray(im.resize((MODEL_SIDE, MODEL_SIDE), Image.BILINEAR), dtype=np.float64)
    lum = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
    lap = (-4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:])
    return {"width": w, "height": h, "min_side": min(w, h), "aspect": w / h,
            "brightness": float(lum.mean()), "contrast": float(lum.std()), "sharpness": float(lap.var())}


def reference_thresholds(cues: list[dict]) -> dict:
    g = lambda k, q: float(np.percentile([c[k] for c in cues], q))  # noqa: E731
    return {"brightness_lo": g("brightness", 10), "brightness_hi": g("brightness", 90),
            "contrast_lo": g("contrast", 10), "sharpness_lo": g("sharpness", 10),
            "min_side_lo": g("min_side", 10), "aspect_lo": g("aspect", 5), "aspect_hi": g("aspect", 95)}


def flags_for(c: dict, t: dict) -> list[str]:
    f = []
    if c["brightness"] < t["brightness_lo"]:
        f.append("gelap")
    if c["brightness"] > t["brightness_hi"]:
        f.append("terang berlebih")
    if c["contrast"] < t["contrast_lo"]:
        f.append("kontras rendah")
    if c["sharpness"] < t["sharpness_lo"]:
        f.append("kabur")
    if c["min_side"] < t["min_side_lo"]:
        f.append("resolusi kecil")
    if c["aspect"] < t["aspect_lo"] or c["aspect"] > t["aspect_hi"]:
        f.append("aspect ratio ekstrem")
    return f


# --------------------------------------------------------------------------- #
# Pemilihan contoh
# --------------------------------------------------------------------------- #
def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[ERROR] {path} tidak ada. Jalankan dulu: python src/evaluate.py --split test --confirm-final")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["prob_dog"] = float(r["prob_dog"])
        r["correct"] = int(r["correct"])
        r["confidence"] = r["prob_dog"] if r["pred"] == "Dog" else 1.0 - r["prob_dog"]
        r["error_type"] = f"{r['label']}->{r['pred']}"
    return rows


def select_examples(errors: list[dict], n_per_type: int, min_total: int = 3) -> list[dict]:
    by_type = {"Cat->Dog": [], "Dog->Cat": []}
    for e in errors:
        by_type[e["error_type"]].append(e)
    chosen = []
    for t in by_type:
        chosen += sorted(by_type[t], key=lambda e: -e["confidence"])[:n_per_type]
    if len(chosen) < min_total:  # tambah dari sisa error bila masih kurang
        rest = [e for e in sorted(errors, key=lambda e: -e["confidence"]) if e not in chosen]
        chosen += rest[: min_total - len(chosen)]
    return sorted(chosen, key=lambda e: -e["confidence"])


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_thumb(src: str, dst: Path, max_side: int = 480) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst, quality=92)


def save_grid(path: Path, examples: list[dict], ncols: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(examples)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, e in zip(axes.ravel(), examples):
        with Image.open(e["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"#{e['rank']}  Aktual: {e['label']} | Prediksi: {e['pred']}\n"
                     f"P(Dog)={e['prob_dog']:.3f}  (yakin {e['confidence']:.1%})", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_report(path: Path, n_test: int, errors: list[dict], examples: list[dict],
                 thr: dict, enrich: list[tuple]) -> None:
    n_err = len(errors)
    types = Counter(e["error_type"] for e in errors)
    mean_conf = float(np.mean([e["confidence"] for e in errors])) if errors else 0.0
    high = sum(1 for e in errors if e["confidence"] >= 0.9)
    L = ["# Analisis Kesalahan Klasifikasi (Test Set)\n",
         f"- Total gambar test: {n_test} | salah klasifikasi: **{n_err}** ({n_err / n_test:.2%})",
         f"- Cat -> Dog (False Positive): **{types.get('Cat->Dog', 0)}** | "
         f"Dog -> Cat (False Negative): **{types.get('Dog->Cat', 0)}**",
         f"- Rata-rata confidence pada prediksi salah: {mean_conf:.1%}; "
         f"kesalahan dengan confidence >= 90%: {high} ({high / max(n_err, 1):.1%})",
         "- Catatan: hasil di bawah adalah HIPOTESIS dari petunjuk teknis otomatis; faktor semantik "
         "(occlusion, pose, background, kemiripan visual, label noise) harus dikonfirmasi dengan melihat gambar.\n",
         "## Faktor teknis yang berlebihan di antara kesalahan\n",
         "Ambang dari gambar yang diklasifikasikan benar (persentil 10/90). Pada gambar benar, "
         "setiap bendera muncul kira-kira 10% (aspect ratio ~10%). Tingkat jauh di atas itu pada "
         "gambar salah menunjukkan faktor yang berkontribusi.\n",
         "| Faktor | Jumlah error berbendera | Persentase error | Acuan gambar benar |", "|---|---:|---:|---:|"]
    for name, cnt, pct in enrich:
        L.append(f"| {name} | {cnt} | {pct:.1%} | ~10% |")
    L += ["", "## Contoh kesalahan (confidence tertinggi per jenis)\n"]
    for e in examples:
        c = e["cues"]
        L += [f"### Contoh #{e['rank']}: aktual **{e['label']}** -> prediksi **{e['pred']}**\n",
              f"![contoh {e['rank']}](misclassified/{e['thumb']})\n",
              f"- File: `{e['path']}`",
              f"- Probabilitas: P(Dog) = {e['prob_dog']:.4f}; confidence pada kelas prediksi = {e['confidence']:.1%}",
              f"- Ukuran asli: {c['width']} x {c['height']} px (aspect {c['aspect']:.2f}); kecerahan {c['brightness']:.0f}/255; "
              f"kontras (std) {c['contrast']:.1f}; ketajaman (var. Laplacian) {c['sharpness']:.0f}",
              f"- Bendera teknis otomatis: {', '.join(e['flags']) if e['flags'] else 'tidak ada'}"]
        hyp = [f"{FLAG_HINTS[f][0]}" for f in e["flags"]]
        if e["confidence"] >= 0.95:
            hyp.append("confidence sangat tinggi: periksa kemungkinan label noise (gambar bukan kucing/anjing, "
                       "label salah, atau banyak objek)")
        L.append("- Hipotesis penyebab (otomatis): " + ("; ".join(hyp) if hyp else
                 "tidak ada petunjuk teknis ekstrem; kemungkinan faktor semantik (pose tidak umum, occlusion, "
                 "background membingungkan, crop kurang baik, kemiripan visual Cat/Dog)"))
        fix = [FLAG_HINTS[f][1] for f in e["flags"]]
        L.append("- Perbaikan/eksperimen yang relevan: " + ("; ".join(fix) if fix else
                 "augmentasi lebih kaya (crop acak, color jitter), model lebih dalam atau resolusi lebih tinggi"))
        L += ["- **Interpretasi setelah melihat gambar (isi manual):** _..._", ""]

    L += ["## Usulan eksperimen lanjutan\n",
          "Berdasarkan faktor yang paling berlebihan di atas (evaluasi pada VALIDATION set, bukan test):\n"]
    top = [x for x in enrich if x[2] > 0.15][:3] or enrich[:1]
    for i, (name, cnt, pct) in enumerate(top, 1):
        L.append(f"{i}. **{name}** ({pct:.1%} dari error): {FLAG_HINTS[name][1]}.")
    L += ["", "Bandingkan setiap eksperimen lewat `train.py --run-name ...` dan `compare_runs.py` memakai "
          "validation loss/accuracy. Test set yang sudah dilihat tidak boleh dipakai untuk memilih eksperimen; "
          "jika ingin melaporkan model baru pada test, itu harus dinyatakan sebagai evaluasi tambahan, bukan "
          "pengganti hasil final.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis kesalahan klasifikasi pada test set.")
    ap.add_argument("--predictions", type=Path, default=FINAL_DIR / "test_predictions.csv")
    ap.add_argument("--out-dir", type=Path, default=FINAL_DIR)
    ap.add_argument("--n-per-type", type=int, default=4)
    ap.add_argument("--ref-size", type=int, default=300, help="Jumlah gambar benar untuk ambang acuan")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_predictions(args.predictions)
    errors = [r for r in rows if r["correct"] == 0]
    correct = [r for r in rows if r["correct"] == 1]
    if not errors:
        sys.exit("[INFO] Tidak ada kesalahan klasifikasi pada test set - tidak ada yang dianalisis.")

    rng = np.random.default_rng(args.seed)
    ref_idx = rng.choice(len(correct), size=min(args.ref_size, len(correct)), replace=False)
    thr = reference_thresholds([image_cues(correct[i]["path"]) for i in ref_idx])

    for e in errors:  # bendera untuk SEMUA error
        e["cues"] = image_cues(e["path"])
        e["flags"] = flags_for(e["cues"], thr)

    examples = select_examples(errors, args.n_per_type)
    mis_dir = args.out_dir / "misclassified"
    mis_dir.mkdir(parents=True, exist_ok=True)
    for rank, e in enumerate(examples, 1):
        e["rank"] = rank
        e["thumb"] = f"{rank:02d}_{e['label']}_as_{e['pred']}.jpg"
        save_thumb(e["path"], mis_dir / e["thumb"])

    flag_counts = Counter(f for e in errors for f in e["flags"])
    enrich = sorted(((name, flag_counts.get(name, 0), flag_counts.get(name, 0) / len(errors))
                     for name in FLAG_HINTS), key=lambda x: -x[2])

    save_grid(args.out_dir / "misclassified_grid.png", examples)
    flat = lambda e: {**{k: e[k] for k in ("path", "label", "pred", "prob_dog", "confidence", "error_type")},  # noqa: E731
                      **e["cues"], "flags": "; ".join(e["flags"])}
    cols = ["path", "label", "pred", "prob_dog", "confidence", "error_type", "width", "height",
            "min_side", "aspect", "brightness", "contrast", "sharpness", "flags"]
    write_csv(args.out_dir / "errors_all.csv", [flat(e) for e in sorted(errors, key=lambda e: -e["confidence"])], cols)
    write_csv(args.out_dir / "misclassified_examples.csv", [{"rank": e["rank"], **flat(e)} for e in examples],
              ["rank"] + cols)
    write_report(args.out_dir / "misclassification_analysis.md", len(rows), errors, examples, thr, enrich)

    types = Counter(e["error_type"] for e in errors)
    print(f"Test: {len(rows)} gambar | salah: {len(errors)} ({len(errors) / len(rows):.2%}) | "
          f"Cat->Dog {types.get('Cat->Dog', 0)}, Dog->Cat {types.get('Dog->Cat', 0)}")
    print("Faktor teknis berlebihan: " + ", ".join(f"{n} {p:.0%}" for n, _, p in enrich[:3]))
    print(f"{len(examples)} contoh dianalisis. Output: {args.out_dir.resolve()}")
    print("Langkah manual: lihat misclassified_grid.png lalu isi 'Interpretasi' pada misclassification_analysis.md")


if __name__ == "__main__":
    main()
