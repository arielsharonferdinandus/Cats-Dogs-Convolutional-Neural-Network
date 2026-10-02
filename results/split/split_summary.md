# Ringkasan Data Split

- Metode: stratified per kelas + group-aware (MD5 duplikat tidak dipisah), numpy.random.default_rng(seed)
- Random seed: **42**
- Rasio target: train 70% / val 15% / test 15%
- Gambar valid masuk split: 24992 dari 24998 (dikeluarkan karena label konflik pada duplikat: 6)
- Kelompok duplikat yang dijaga tetap satu split: 26

| Split | Cat | Dog | Total | Proporsi aktual | Proporsi Cat |
|---|---:|---:|---:|---:|---:|
| train | 8747 | 8746 | 17493 | 69.99% | 50.00% |
| val | 1875 | 1875 | 3750 | 15.00% | 50.00% |
| test | 1875 | 1874 | 3749 | 15.00% | 50.01% |

## Pemeriksaan integritas

- Overlap path antar split: 0
- Overlap MD5 (konten identik) antar split: 0

## SHA-256 file split

- `train.csv`: `43a85d85c4aad2c66400ba52f31ae57ac238b523248ff7640b95715bede16386`
- `val.csv`: `1c5aabe8e0591b03e08a81be04203c4b602908dd646b544a2fef17e725a95c6c`
- `test.csv`: `8bfaba605a046ffa9b88f4bee5911a8fab63e3ca072cd372765030861a7d3945`
