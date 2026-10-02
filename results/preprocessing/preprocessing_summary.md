# Ringkasan Preprocessing

- Target image size: **128 x 128** (resize langsung, bilinear + antialias, aspect ratio tidak dijaga)
- Jumlah channel: **3** (semua gambar dikonversi ke RGB)
- Normalisasi: skala piksel ke [0, 1], lalu `(x - mean) / std` per channel
  - mean (R,G,B) = [0.488141, 0.454966, 0.416358]
  - std  (R,G,B) = [0.257589, 0.250695, 0.253535]
  - dihitung dari **training set saja** (17493 gambar) - tidak ada informasi val/test
- Encoding label: **Cat = 0, Dog = 1**, float32; model memiliki 1 logit, `sigmoid(logit) = P(Dog)`;
  loss `BCEWithLogitsLoss`

| Split | Pipeline |
|---|---|
| Train | Resize(128) -> RandomHorizontalFlip(0.5) -> RandomRotation(10 derajat) -> ToTensor -> Normalize |
| Validation | Resize(128) -> ToTensor -> Normalize |
| Test | Resize(128) -> ToTensor -> Normalize (identik dengan validation) |

Augmentasi hanya pada training. Statistik normalisasi yang sama dipakai untuk ketiga split.
Contoh hasil: `sample_batch.png`.
