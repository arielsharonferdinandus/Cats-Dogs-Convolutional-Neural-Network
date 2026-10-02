# Perhitungan Manual Output Shape & Parameter

Rumus: `Output = floor((N + 2P - K) / S) + 1`, `Param conv = (K x K x C_in x C_out) + C_out`

## A. Contoh wajib task (32 x 32 x 3)

- Input: 32 x 32 x 3; 16 filter, kernel 3x3, padding 1, stride 1
- Output height = floor((32 + 2*1 - 3) / 1) + 1 = **32**
- Output width  = **32**
- Output channels = jumlah filter = **16**
- Parameter = (3x3x3x16) + 16 = 432 + 16 = **448**
- Output shape: **32 x 32 x 16**

## B. Model utama (input 128 x 128 x 3)

| Layer | Tipe | Output shape (H x W x C) | Parameter | Perhitungan |
|---|---|---|---:|---|
| input | Input | 128 x 128 x 3 | 0 | 128 x 128 x 3 |
| conv1 | Conv2d | 128 x 128 x 32 | 896 | out = floor((128+2*1-3)/1)+1 = 128; params = (3*3*3*32)+32 = 896 |
| bn1 | BatchNorm2d | 128 x 128 x 32 | 64 | 2*32 = 64 (gamma, beta) |
| relu1 | ReLU | 128 x 128 x 32 | 0 | elementwise, shape tetap |
| pool1 | MaxPool2d | 64 x 64 x 32 | 0 | out = floor((128+0-2)/2)+1 = 64 |
| conv2 | Conv2d | 64 x 64 x 64 | 18,496 | out = floor((64+2*1-3)/1)+1 = 64; params = (3*3*32*64)+64 = 18496 |
| bn2 | BatchNorm2d | 64 x 64 x 64 | 128 | 2*64 = 128 (gamma, beta) |
| relu2 | ReLU | 64 x 64 x 64 | 0 | elementwise, shape tetap |
| pool2 | MaxPool2d | 32 x 32 x 64 | 0 | out = floor((64+0-2)/2)+1 = 32 |
| conv3 | Conv2d | 32 x 32 x 128 | 73,856 | out = floor((32+2*1-3)/1)+1 = 32; params = (3*3*64*128)+128 = 73856 |
| bn3 | BatchNorm2d | 32 x 32 x 128 | 256 | 2*128 = 256 (gamma, beta) |
| relu3 | ReLU | 32 x 32 x 128 | 0 | elementwise, shape tetap |
| pool3 | MaxPool2d | 16 x 16 x 128 | 0 | out = floor((32+0-2)/2)+1 = 16 |
| conv4 | Conv2d | 16 x 16 x 256 | 295,168 | out = floor((16+2*1-3)/1)+1 = 16; params = (3*3*128*256)+256 = 295168 |
| bn4 | BatchNorm2d | 16 x 16 x 256 | 512 | 2*256 = 512 (gamma, beta) |
| relu4 | ReLU | 16 x 16 x 256 | 0 | elementwise, shape tetap |
| pool4 | MaxPool2d | 8 x 8 x 256 | 0 | out = floor((16+0-2)/2)+1 = 8 |
| gap | AdaptiveAvgPool2d | 1 x 1 x 256 | 0 | rata-rata spasial 8x8 -> 1x1 |
| flatten | Flatten | 1 x 1 x 256 | 0 | 1x1x256 -> vektor 256 |
| dropout | Dropout | 1 x 1 x 256 | 0 | p=0.3, tidak ada parameter |
| fc | Linear | 1 x 1 x 1 | 257 | (256*1)+1 = 257 -> 1 logit; P(Dog)=sigmoid(logit) |

- Total parameter conv: **388,416**
- Total parameter model (trainable): **389,633**
