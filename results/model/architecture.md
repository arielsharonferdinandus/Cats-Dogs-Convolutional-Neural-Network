# Arsitektur CNN

```
Input  128x128x3
  |
  v  Block 1: Conv2d(3x3, 32 filter, pad=1, stride=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
     conv out 128 x 128 x 32 -> pool out 64 x 64 x 32
  |
  v  Block 2: Conv2d(3x3, 64 filter, pad=1, stride=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
     conv out 64 x 64 x 64 -> pool out 32 x 32 x 64
  |
  v  Block 3: Conv2d(3x3, 128 filter, pad=1, stride=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
     conv out 32 x 32 x 128 -> pool out 16 x 16 x 128
  |
  v  Block 4: Conv2d(3x3, 256 filter, pad=1, stride=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
     conv out 16 x 16 x 256 -> pool out 8 x 8 x 256
  |
  v  GlobalAvgPool -> Flatten (256)
  |
  v  Dropout(p=0.3)
  |
  v  Linear(256 -> 1)  = 1 logit
  |
  v  sigmoid -> P(Dog)   (Cat=0, Dog=1)
```
