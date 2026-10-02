# Microsoft Cats vs Dogs - Klasifikasi Biner dengan CNN

## Project overview
CNN kecil (dilatih dari awal, tanpa transfer learning) untuk klasifikasi biner **Cat vs Dog**
menggunakan PyTorch. Proyek menekankan **reproducibility** dan **isolasi test set**: dataset
divalidasi, di-split dengan seed tercatat, model dipilih hanya dari validation set, konfigurasi
final dikunci, lalu test set dievaluasi **sekali**.

## Dataset
Microsoft Cats vs Dogs (Kaggle): https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
Diunduh lewat library `kagglehub` (butuh akun & token Kaggle).

## Environment
- Python **3.10+** (kode memakai type hint modern; dikembangkan untuk Python 3.12)
- Dependency: `kagglehub`, `numpy`, `pillow`, `matplotlib`, `torch`, `torchvision` (lihat `requirements.txt`)
- Device: **otomatis** CUDA -> MPS (Apple Silicon) -> CPU. Device yang dipakai dicatat di
  `results/runs/<run>/config.json` dan `train_summary.md`. Dapat dipaksa dengan `--device cpu|cuda|mps`.

## Instalasi
Dengan **uv** (disarankan, Ubuntu):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # jika uv belum terpasang
uv sync                                            # membuat .venv dan memasang dependency dari pyproject.toml
```
Tanpa GPU, aktifkan blok PyTorch CPU-only yang dikomentari di `pyproject.toml` sebelum `uv sync`
(build CUDA bawaan PyPI di Linux berukuran beberapa GB).

Alternatif tanpa uv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kredensial Kaggle (pilih salah satu): `~/.kaggle/kaggle.json` (`chmod 600`), env
`KAGGLE_USERNAME` + `KAGGLE_KEY`, atau `KAGGLE_API_TOKEN`; atau `download_dataset.py --login`.

Jika dataset sudah ada di mesin, letakkan/symlink sebagai `data/raw/PetImages/{Cat,Dog}` lalu
gunakan `--skip-download` (struktur: `PetImages/Cat/0.jpg ... N.jpg`, `PetImages/Dog/0.jpg ... N.jpg`).

## Cara menjalankan
Semua lewat satu pintu masuk `main.py` (boleh dijalankan dari direktori mana pun; tiap perintah memanggil
script di `src/`, dan opsi tambahan diteruskan apa adanya):

```bash
uv run main.py status                      # langkah apa yang sudah selesai
uv run main.py setup                       # download -> inspect -> split -> preprocess -> model -> calc -> verify
uv run main.py setup --skip-download       # jika dataset sudah ada di data/raw/PetImages
uv run main.py all                         # setup + training baseline (BERHENTI sebelum test set)

# Eksperimen (hanya train & validation)
uv run main.py train --run-name lr3e-4 --lr 3e-4
uv run main.py train --run-name no_aug --no-augment
uv run main.py compare                                   # tabel perbandingan (urut val loss)
uv run main.py eval-val --run-name baseline              # precision/recall/F1 pada validation

# Kunci konfigurasi final, lalu evaluasi akhir SEKALI
uv run main.py compare --select <run> --reason "val loss terendah"
uv run main.py eval-test --confirm-final
uv run main.py analyze                                   # misclassification analysis

# Opsional: demo verifikasi data-pipeline dengan fault injection (dataset mini sintetis)
uv run main.py verify --demo
```
Setiap langkah juga bisa dijalankan langsung, mis. `uv run python src/split_data.py --seed 42`.
`uv run main.py <perintah> --help` menampilkan opsi script terkait.

`all` sengaja tidak memuat pemilihan konfigurasi maupun evaluasi test: kedua langkah itu harus
dilakukan manual agar test set tetap unseen sampai konfigurasi final ditentukan dari validation set.

## Struktur project
```
catsdogs_cnn/
├── main.py                  # pintu masuk: uv run main.py <perintah>
├── pyproject.toml           # dependency untuk uv (requirements.txt = alternatif pip)
├── README.md
├── requirements.txt
├── src/
│   ├── config.py            # konfigurasi bersama (path, label, ModelConfig, seed)
│   ├── download_dataset.py  # unduh via kagglehub
│   ├── inspect_dataset.py   # inspeksi & validasi dataset
│   ├── split_data.py        # split reproducible, group-aware (MD5), stratified
│   ├── preprocessing.py     # transform, Dataset, statistik normalisasi, penjaga test set
│   ├── model.py             # SmallCNN + layer_report
│   ├── manual_calc.py       # hitung manual output shape & parameter
│   ├── utils.py             # seed & device
│   ├── train.py             # training + validasi, kurva, checkpoint
│   ├── compare_runs.py      # bandingkan run (val) & kunci konfigurasi final
│   ├── metrics.py           # accuracy, precision, recall, F1, AUC, confusion matrix
│   ├── evaluate.py          # evaluasi val / test (test: sekali, dijaga & dicatat)
│   ├── error_analysis.py    # analisis kesalahan klasifikasi
│   └── verify_pipeline.py   # verifikasi pipeline (label, file rusak, shape)
├── data/                    # (dibuat otomatis, tidak di-commit)
│   ├── kagglehub_cache/  raw/PetImages/{Cat,Dog}  splits/{train,val,test}.csv  split_manifest.json  norm_stats.json
└── results/                 # (dibuat otomatis)
    ├── inspection/  split/  preprocessing/  model/  verification/
    ├── runs/<run>/          # config.json, history.csv, training_curves.png, best_model.pt, val_metrics.*
    └── final/               # selected_config.json, test_access_log.jsonl, test_metrics.*, confusion_matrix_test.png,
                             # misclassification_analysis.md, misclassified_grid.png
```

## Konfigurasi penting
| Item | Nilai default |
|---|---|
| Input | 128 x 128 x 3 (resize langsung, RGB) |
| Normalisasi | skala [0,1] lalu `(x-mean)/std` per channel; mean/std dari **train saja** |
| Label | Cat = 0, Dog = 1; 1 logit, `sigmoid` = P(Dog) |
| Arsitektur | 4 x [Conv3x3(pad 1) -> BatchNorm -> ReLU -> MaxPool2], filter 32/64/128/256, GAP, Dropout 0.3, Linear(1) |
| Loss / Optimizer | `BCEWithLogitsLoss` / Adam, LR 1e-3 (konstan) |
| Batch size / Epoch | 64 / maks 20, early stopping (patience 5, pada val loss) |
| Augmentasi | hanya train: horizontal flip, rotasi 10 derajat |
| Threshold | 0.5 (tetap) |
| Split | 70 / 15 / 15, stratified per kelas, duplikat MD5 tidak dipisah |

## Random seed
**42** - dipakai oleh `split_data.py` (split) dan `train.py` (Python, NumPy, PyTorch, DataLoader;
cuDNN deterministik). Seed dicatat di `split_manifest.json` dan `config.json` tiap run.
Reproducibility bit-per-bit antar GPU/versi PyTorch yang berbeda tidak dijamin.

## Isolasi test set (ringkas)
1. `train.py` hanya membaca `train.csv` dan `val.csv`; `read_split_csv("test")` menolak tanpa `allow_test=True`.
2. Semua pemilihan arsitektur/optimizer/LR/batch/epoch/preprocessing memakai **validation set**.
3. `compare_runs.py --select` mengunci run + SHA-256 checkpoint **sebelum** test dibuka.
4. `evaluate.py --split test` menolak jika kunci tidak ada, checkpoint berubah, atau test sudah pernah
   dievaluasi; setiap akses dicatat di `results/final/test_access_log.jsonl`.
