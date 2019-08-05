# coding: utf-8

import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import os

import torch
import torchvision
from torchvision import models, transforms

# Check PyTorch version.
print('PyTorch Version: ', torch.__version__)
print('Torchvision Version: ', torchvision.__version__)

# Load pre-trained VGG16 model.

# Generate instance of VGG16 model
net = models.vgg16(pretrained=True)  # Use pre-trained parameters.
net.eval()  # Set inference mode.

# Output network structure of a model.
print(net)


# pre-processing class of input images.
class BaseTransform():
	"""
	Resize image size and normalize color.

	Attributes
	----------
	resize: int
		size to resize a image
	mean: (R, G, B)
		mean of each channel
	std: (R, G, B)
		standard deviation of each channel
	"""
	def __init__(self, resize, mean, std):
		self.base_transform = transforms.ComPOse([
			transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
			transforms.CenterCrop(resize),   # 画像中央をresize x resizeで切り取り
			transforms.ToTensor(),  # Torchテンソルに変換
			transforms.Normalize(mean, std)  # 色情報の標準化
		])


	def __call__(self, img):
		return self.base_transform(img)


# 画像前処理の動作を確認

# 1. 画像読み込み
image_file_path = '../data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

# 2. 元画像の表示
plt.imshow(img)
plt.show()







