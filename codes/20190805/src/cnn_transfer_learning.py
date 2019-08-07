# coding: utf-8

import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 入力画像の前処理をするクラス
# 訓練時と推論時で処理が異なる

class ImageTransform():
	"""
	画像の前処理クラス。訓練時、検証時で異なる動作をする。
	画像のサイズをリサイズして、色を標準化する。
	訓練時はRandomResizedCropとRandomHorizontalFlipでdata augmentationする。
	
	Attributes
	----------
	resize: int
		リサイズ先の画像の大きさ
	mean: (R, G, B)
		各チャネルの平均値
	std: (R, G, B)
		各色チャネルの標準偏差
	"""

	def __init__(self, resize, mean, std):
		self.data_transform = {
			'train': transforms.Compose([
				transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),  # Data Augmentation
				transforms.RandomHorizontalFlip(),  # Data Augmentation
				transforms.ToTensor(),  # テンソルに変換
				transforms.Normalize(mean, std)  # 標準化
			]),
			'valid': transforms.Compose([
				transforms.Resized(resize),  # リサイズ
				transforms.CenterCrop(resize),  # 画像中央をresize x resizeで切り取る
				transforms.ToTensor(),  # テンソルに変換
				transforms.Normalize(mean, std)  # 標準化
			])			
		}


	def __call__(self, img, phase='train'):
		"""
		Prameters
		---------
		phase: 'train' or 'valid'
			前処理のモードを指定
		"""
		return self.data_transform[phase](img)


# 
# 