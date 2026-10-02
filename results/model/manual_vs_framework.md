# Perbandingan Manual vs Model Summary (PyTorch)

Status keseluruhan: **SEMUA COCOK**

| Layer | Shape manual | Shape framework | Param manual | Param framework | Cocok |
|---|---|---|---:|---:|:---:|
| conv1 | 128 x 128 x 32 | 128 x 128 x 32 | 896 | 896 | ya |
| bn1 | 128 x 128 x 32 | 128 x 128 x 32 | 64 | 64 | ya |
| relu1 | 128 x 128 x 32 | 128 x 128 x 32 | 0 | 0 | ya |
| pool1 | 64 x 64 x 32 | 64 x 64 x 32 | 0 | 0 | ya |
| conv2 | 64 x 64 x 64 | 64 x 64 x 64 | 18,496 | 18,496 | ya |
| bn2 | 64 x 64 x 64 | 64 x 64 x 64 | 128 | 128 | ya |
| relu2 | 64 x 64 x 64 | 64 x 64 x 64 | 0 | 0 | ya |
| pool2 | 32 x 32 x 64 | 32 x 32 x 64 | 0 | 0 | ya |
| conv3 | 32 x 32 x 128 | 32 x 32 x 128 | 73,856 | 73,856 | ya |
| bn3 | 32 x 32 x 128 | 32 x 32 x 128 | 256 | 256 | ya |
| relu3 | 32 x 32 x 128 | 32 x 32 x 128 | 0 | 0 | ya |
| pool3 | 16 x 16 x 128 | 16 x 16 x 128 | 0 | 0 | ya |
| conv4 | 16 x 16 x 256 | 16 x 16 x 256 | 295,168 | 295,168 | ya |
| bn4 | 16 x 16 x 256 | 16 x 16 x 256 | 512 | 512 | ya |
| relu4 | 16 x 16 x 256 | 16 x 16 x 256 | 0 | 0 | ya |
| pool4 | 8 x 8 x 256 | 8 x 8 x 256 | 0 | 0 | ya |
| gap | 1 x 1 x 256 | 1 x 1 x 256 | 0 | 0 | ya |
| flatten | 1 x 1 x 256 | 1 x 1 x 256 | 0 | 0 | ya |
| dropout | 1 x 1 x 256 | 1 x 1 x 256 | 0 | 0 | ya |
| fc | 1 x 1 x 1 | 1 x 1 x 1 | 257 | 257 | ya |

Total parameter manual: **389,633** | framework: **389,633**

## Contoh task 32x32x3 -> 16 filter
- Manual : 32 x 32 x 16, 448 parameter
- PyTorch: 32 x 32 x 16 (HxWxC), 448 parameter
- Cocok: **ya**
