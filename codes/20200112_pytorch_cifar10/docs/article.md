
```
$ python .\train.py
Files already downloaded and verified
Files already downloaded and verified
batch:  10/391, train acc: 0.176831, train loss: 0.000512
batch:  20/391, train acc: 0.208884, train loss: 0.000931
batch:  30/391, train acc: 0.214069, train loss: 0.001356
batch:  40/391, train acc: 0.227074, train loss: 0.001739
batch:  50/391, train acc: 0.240399, train loss: 0.002111
batch:  60/391, train acc: 0.251741, train loss: 0.002479
batch:  70/391, train acc: 0.263337, train loss: 0.002838
batch:  80/391, train acc: 0.272557, train loss: 0.003192
batch:  90/391, train acc: 0.281386, train loss: 0.003542
batch: 100/391, train acc: 0.289942, train loss: 0.003877
batch: 110/391, train acc: 0.296704, train loss: 0.004221
batch: 120/391, train acc: 0.304817, train loss: 0.004563
batch: 130/391, train acc: 0.312345, train loss: 0.004887
batch: 140/391, train acc: 0.318597, train loss: 0.005206
batch: 150/391, train acc: 0.324684, train loss: 0.005527
batch: 160/391, train acc: 0.33013 , train loss: 0.005852
batch: 170/391, train acc: 0.337387, train loss: 0.006161
batch: 180/391, train acc: 0.342946, train loss: 0.006469
batch: 190/391, train acc: 0.345693, train loss: 0.006794
batch: 200/391, train acc: 0.348746, train loss: 0.007111
batch: 210/391, train acc: 0.353752, train loss: 0.00741
batch: 220/391, train acc: 0.358905, train loss: 0.007701
batch: 230/391, train acc: 0.363425, train loss: 0.008002
batch: 240/391, train acc: 0.367813, train loss: 0.00829
batch: 250/391, train acc: 0.371961, train loss: 0.008572
batch: 260/391, train acc: 0.375724, train loss: 0.008852
batch: 270/391, train acc: 0.380606, train loss: 0.009122
batch: 280/391, train acc: 0.385157, train loss: 0.0094
batch: 290/391, train acc: 0.389324, train loss: 0.00967
batch: 300/391, train acc: 0.393095, train loss: 0.009946
batch: 310/391, train acc: 0.396681, train loss: 0.010216
batch: 320/391, train acc: 0.400752, train loss: 0.01048
batch: 330/391, train acc: 0.404503, train loss: 0.010741
batch: 340/391, train acc: 0.408347, train loss: 0.010993
batch: 350/391, train acc: 0.411961, train loss: 0.011244
batch: 360/391, train acc: 0.415064, train loss: 0.011502
batch: 370/391, train acc: 0.418225, train loss: 0.011753
batch: 380/391, train acc: 0.421141, train loss: 0.012009
batch: 390/391, train acc: 0.424302, train loss: 0.012254
epoch:   1, train acc: 0.424302, train loss: 0.012254, test acc: 0.539407, test loss: 0.012638
Saved a model checkpoint at ../experiments/models/checkpoints/CIFAR10_ResNet18_epoch=1.pth
```

```
$ python .\test.py
Files already downloaded and verified
Files already downloaded and verified
Loaded a model from ../experiments/models/CIFAR10_ResNet18_epoch=10.pth
test acc: 0.779172, test loss: 0.006796
```

## 参考文献
- (Train CIFAR10 with PyTorch - github/kuangliu)[https://github.com/kuangliu/pytorch-cifar]