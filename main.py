#!/usr/bin/env python3
"""
main.py - satu pintu masuk untuk seluruh pipeline Cats vs Dogs.

Setiap subcommand hanya memanggil script di src/ (subprocess), jadi script lama tetap bisa
dijalankan sendiri dan opsi aslinya diteruskan apa adanya. main.py juga selalu berpindah ke
folder proyek, sehingga boleh dijalankan dari direktori mana pun.

Contoh (dengan uv):
  uv run main.py status
  uv run main.py setup                       # download -> inspect -> split -> preprocess -> model -> calc -> verify
  uv run main.py setup --skip-download       # jika dataset sudah ada di data/raw/PetImages
  uv run main.py train --run-name lr3e-4 --lr 3e-4
  uv run main.py compare --select baseline --reason "val loss terendah"
  uv run main.py eval-test --confirm-final   # EVALUASI AKHIR, sekali saja
  uv run main.py analyze

'all' = setup + train baseline. Sengaja BERHENTI sebelum pemilihan konfigurasi & test set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# nama subcommand -> (script, argumen tetap, deskripsi)
STEPS: dict[str, tuple[str, list[str], str]] = {
    "download":   ("download_dataset.py", [], "Unduh dataset via kagglehub"),
    "inspect":    ("inspect_dataset.py", [], "Inspeksi & validasi dataset (file rusak, ukuran, kelas)"),
    "split":      ("split_data.py", [], "Split train/val/test reproducible"),
    "preprocess": ("preprocessing.py", [], "Statistik normalisasi (train saja) + cek batch"),
    "model":      ("model.py", [], "Cetak model summary"),
    "calc":       ("manual_calc.py", [], "Hitung manual output shape/parameter vs model summary"),
    "verify":     ("verify_pipeline.py", [], "Verifikasi data-pipeline (tambah --demo untuk fault injection)"),
    "train":      ("train.py", [], "Training + validasi (opsi: --run-name --lr --epochs ...)"),
    "compare":    ("compare_runs.py", [], "Bandingkan run (val); --select RUN untuk mengunci konfigurasi final"),
    "eval-val":   ("evaluate.py", ["--split", "val"], "Evaluasi pada validation set"),
    "eval-test":  ("evaluate.py", ["--split", "test"], "EVALUASI AKHIR test set (wajib --confirm-final, sekali saja)"),
    "analyze":    ("error_analysis.py", [], "Analisis kesalahan klasifikasi (setelah eval-test)"),
}

GROUPS: dict[str, list[str]] = {
    "setup": ["download", "inspect", "split", "preprocess", "model", "calc", "verify"],
    "all":   ["download", "inspect", "split", "preprocess", "model", "calc", "verify", "train"],
}


def run_step(name: str, extra: list[str]) -> int:
    script, fixed, _ = STEPS[name]
    cmd = [sys.executable, str(Path("src") / script), *fixed, *extra]
    print(f"\n{'=' * 70}\n>>> [{name}] {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    return subprocess.run(cmd).returncode


def run_group(group: str, skip_download: bool) -> int:
    for name in GROUPS[group]:
        if skip_download and name == "download":
            print("\n[skip] download (--skip-download); memakai data/raw/PetImages yang sudah ada")
            continue
        code = run_step(name, [])
        if code != 0:
            print(f"\n[BERHENTI] Langkah '{name}' gagal (exit code {code}). Perbaiki lalu jalankan ulang.")
            return code
    if group == "all":
        print("\n[SELESAI] Pipeline sampai training baseline. Berhenti sebelum test set.\n"
              "Langkah berikutnya (manual, disengaja):\n"
              "  1. uv run main.py train --run-name <variasi> ...   # eksperimen lain, dinilai lewat validation\n"
              "  2. uv run main.py compare                          # lihat tabel\n"
              "  3. uv run main.py compare --select <run> --reason \"...\"   # kunci konfigurasi final\n"
              "  4. uv run main.py eval-test --confirm-final        # evaluasi akhir SEKALI\n"
              "  5. uv run main.py analyze")
    return 0


def status() -> None:
    checks = [
        ("Dataset di data/raw/PetImages", "data/raw/PetImages/Cat"),
        ("Inspeksi dataset", "results/inspection/valid_files.csv"),
        ("Split train/val/test", "data/splits/split_manifest.json"),
        ("Statistik normalisasi", "data/splits/norm_stats.json"),
        ("Hitungan manual vs framework", "results/model/manual_vs_framework.md"),
        ("Verifikasi pipeline (data asli)", "results/verification/pipeline_check.md"),
        ("Training (>=1 run selesai)", "results/runs"),
        ("Konfigurasi final TERKUNCI", "results/final/selected_config.json"),
        ("Test set sudah dievaluasi", "results/final/test_access_log.jsonl"),
        ("Analisis kesalahan", "results/final/misclassification_analysis.md"),
    ]
    print(f"Proyek: {ROOT}\n")
    for label, rel in checks:
        p = ROOT / rel
        ok = any(p.glob("*/config.json")) if rel == "results/runs" and p.is_dir() else p.exists()
        print(f"  [{'x' if ok else ' '}] {label}")
    print("\nPerintah: " + ", ".join(list(STEPS) + list(GROUPS) + ["status"]))


def usage() -> str:
    L = ["Pemakaian: uv run main.py <perintah> [opsi untuk script terkait]\n", "Perintah:"]
    for k, (_, fixed, desc) in STEPS.items():
        L.append(f"  {k:<11} {desc}")
    L += ["  setup       Jalankan: " + " -> ".join(GROUPS["setup"]) + "  (opsi: --skip-download)",
          "  all         setup + train baseline (berhenti sebelum test set)",
          "  status      Tampilkan langkah yang sudah selesai",
          "\nOpsi apa pun setelah nama perintah diteruskan ke script-nya, mis. 'train --lr 3e-4', "
          "'train --help'."]
    return "\n".join(L)


def main() -> None:
    os.chdir(ROOT)  # path di CSV relatif terhadap root proyek
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(usage())
        return
    cmd, extra = args[0], args[1:]

    if cmd == "status":
        status()
    elif cmd in GROUPS:
        skip = "--skip-download" in extra
        unknown = [a for a in extra if a != "--skip-download"]
        if unknown:
            sys.exit(f"[ERROR] '{cmd}' hanya menerima --skip-download. Untuk opsi lain jalankan langkahnya satu per satu.")
        sys.exit(run_group(cmd, skip))
    elif cmd in STEPS:
        sys.exit(run_step(cmd, extra))
    else:
        print(f"Perintah tidak dikenal: '{cmd}'\n\n{usage()}")
        sys.exit(2)


if __name__ == "__main__":
    main()
