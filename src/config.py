"""
config.py
---------
Satu sumber kebenaran (single source of truth) untuk konfigurasi proyek.
Dipakai oleh preprocessing.py, model.py, manual_calc.py, train.py, dst.
Modul ini sengaja TIDAK mengimpor torch agar manual_calc.py bisa jalan tanpa torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ----------------------------- path ----------------------------------------
SPLIT_DIR = Path("data/splits")
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"
NORM_STATS_PATH = SPLIT_DIR / "norm_stats.json"
RESULTS_DIR = Path("results")
FINAL_DIR = RESULTS_DIR / "final"           # hasil akhir + kunci konfigurasi + log akses test
SELECTION_LOCK = FINAL_DIR / "selected_config.json"
TEST_ACCESS_LOG = FINAL_DIR / "test_access_log.jsonl"

# ----------------------------- label ---------------------------------------
# Binary classification dengan 1 logit: sigmoid(logit) = P(Dog)
CLASS_TO_IDX = {"Cat": 0, "Dog": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ----------------------------- reproducibility -----------------------------
SEED = 42  # harus sama dengan seed split_data.py agar eksperimen konsisten


# ----------------------------- model ---------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 128                       # tinggi = lebar (px)
    in_channels: int = 3                        # RGB
    conv_channels: tuple = (32, 64, 128, 256)   # jumlah filter tiap conv block
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    pool_size: int = 2
    pool_stride: int = 2
    dropout: float = 0.3


DEFAULT_MODEL = ModelConfig()
