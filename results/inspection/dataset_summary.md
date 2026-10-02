# Ringkasan Dataset

- Sumber: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset (versi 1, fingerprint a1edd2d9d4b8...)
- Waktu inspeksi (UTC): 2026-10-02T01:04:23+00:00
- Root dataset: `/home/joker/kuliah/s5/Deep2/DL_Sesi6/data/raw/PetImages`
- Total file ditemukan (semua jenis): **25002**
- Gambar valid: **24998** | korup: **2** | non-gambar: **2**

## Struktur folder

```
PetImages/
├── Cat/   (12501 file)
├── Dog/   (12501 file)
```

## Jumlah per kelas

| Kelas | Total file | Valid | Korup | Non-gambar | Proporsi valid |
|---|---:|---:|---:|---:|---:|
| Cat | 12501 | 12499 | 1 | 1 | 50.00% |
| Dog | 12501 | 12499 | 1 | 1 | 50.00% |

Rasio kelas (Cat : Dog, valid) = 1.0000 : 1

## Format file

| Kelas | Ekstensi | Format hasil decode | Mode (channel) |
|---|---|---|---|
| Cat | {'.jpg': 12500, '.db': 1} | {'JPEG': 12405, 'BMP': 71, 'GIF': 22, 'PSD': 1} | {'RGB (3ch)': 12470, 'P (1ch)': 26, 'L (1ch)': 3} |
| Dog | {'.jpg': 12500, '.db': 1} | {'JPEG': 12364, 'GIF': 29, 'BMP': 102, 'PNG': 4} | {'RGB (3ch)': 12461, 'P (1ch)': 31, 'RGBA (4ch)': 2, 'L (1ch)': 2, 'CMYK (4ch)': 3} |

File dengan ekstensi tidak cocok dengan format asli: **229**

## Statistik ukuran gambar (hanya gambar valid)

| Cakupan | Metrik | Min | Max | Mean | Median | Std | P5 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Semua | width | 4 | 500 | 404.45 | 448 | 109 | 192 | 500 |
| Semua | height | 4 | 500 | 360.99 | 375 | 97 | 181 | 500 |
| Semua | aspect_ratio_w_over_h | 0.306 | 5.93939 | 1.16 | 1.26904 | 0.29 | 0.668 | 1.50602 |
| Cat | width | 4 | 500 | 410.84 | 470 | 107.91 | 197.9 | 500 |
| Cat | height | 4 | 500 | 356.94 | 375 | 96.52 | 180 | 500 |
| Cat | aspect_ratio_w_over_h | 0.35461 | 3.71795 | 1.19 | 1.33333 | 0.29 | 0.69 | 1.5528 |
| Dog | width | 42 | 500 | 398.06 | 423 | 109.7 | 186 | 500 |
| Dog | height | 33 | 500 | 365.04 | 375 | 97.31 | 187 | 500 |
| Dog | aspect_ratio_w_over_h | 0.306 | 5.93939 | 1.12 | 1.19048 | 0.29 | 0.666 | 1.5015 |

Channel (jumlah band) pada gambar valid: {3: 24931, 1: 62, 4: 5}

## File korup / tidak dapat dibaca

Jumlah: **2** (daftar lengkap: `corrupt_files.csv`)

- `Cat/666.jpg` - file kosong (0 byte)
- `Dog/11702.jpg` - file kosong (0 byte)

## Peringatan lain

- Gambar dengan peringatan decoder (mis. EXIF rusak) tetapi tetap terbaca: 1
- Kelompok duplikat persis (MD5): 28 (total file ikut kelompok: 58)
- Gambar sangat kecil (< 32px): 1

## Keputusan untuk tahap berikutnya

- Hanya file pada `valid_files.csv` yang boleh dipakai untuk split/training/validation/test.
- Semua gambar akan dikonversi ke RGB 3 channel pada preprocessing (menangani grayscale/RGBA/palette).
- Duplikat persis sebaiknya dikelompokkan pada split agar tidak bocor antar train/val/test (kolom `md5`).
