from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(dataset_dir, img_show=False):
	# Dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
	train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	if img_show == True:
		show_data(train_loader)

	return train_loader, test_loader, classes


def show_data(data_loader):
	images, labels = iter(data_loader).next()  # data_loader のミニバッチの image を取得
	img = torchvision.utils.make_grid(images, nrow=16, padding=1)  # nrom*nrom のタイル形状の画像を作る
	plt.ion()
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 画像を matplotlib 用に変換
	plt.draw()
	plt.pause(3)  # Display an image for three seconds.
	plt.close()



