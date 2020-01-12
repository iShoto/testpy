# PyTorch + CIFAR10 + ResNet18
This library makes a ResNet18 model classifing CIFAR10 with PyTorch.

# Feature
- Training and test ResNet18 classifing CIFAR10
- Calculating training and test accurary and loss


# Requirement
- python 3.6.8
- pytorch 1.1.0
- torchvision 0.3.0
- scikit-learn 0.20.3
- cuda 9.0
- cudnn 7.1 


# Installation
 
```sh
git clone https://github.com/iShoto/testpy.git
```


# Usage
 
```sh
$ cd testpy/codes/20200112_pytorch_cifar10/src/
$ python train.py
Files already downloaded and verified
Files already downloaded and verified
batch:  10/391, train acc: 0.176831, train loss: 0.000512
batch:  20/391, train acc: 0.208884, train loss: 0.000931
batch:  30/391, train acc: 0.214069, train loss: 0.001356
...
batch: 390/391, train acc: 0.424302, train loss: 0.012254
epoch:   1, train acc: 0.424302, train loss: 0.012254, test acc: 0.539407, test loss: 0.012638
Saved a model checkpoint at ../experiments/models/checkpoints/CIFAR10_ResNet18_epoch=1.pth

$ python test.py
Files already downloaded and verified
Files already downloaded and verified
Loaded a model from ../experiments/models/CIFAR10_ResNet18_epoch=10.pth
test acc: 0.779172, test loss: 0.006796
```


# Author
Shoto I.


# License
[MIT license](https://github.com/iShoto/testpy/blob/master/LICENSE).
