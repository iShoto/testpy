import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import Function
import torchvision

import matplotlib.pyplot as plt
import argparse
from tqdm import trange
import numpy as np
from sklearn.metrics import classification_report

from losses import CenterLoss
from mnistnet import Net

# cf. https://cpp-learning.com/center-loss/


def load_dataset(dataset_dir, batch_size):
	# Dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
	
	return test_loader


def show_data(train_loader):
	images, labels = iter(train_loader).next()  # train_loader のミニバッチの image を取得
	img = torchvision.utils.make_grid(images, nrow=12, padding=1)  # nrom*nrom のタイル形状の画像を作る
	plt.ion()
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 画像を matplotlib 用に変換
	plt.draw()
	plt.pause(3)  # Display an image for three seconds.


def main():
	calc_accuracy()


def calc_accuracy():
	args = parse_args()

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	model = model.eval()
	print(model)

	# Dataset
	test_loader = load_dataset(args.dataset_dir, batch_size=128)
	show_data(test_loader)

	# Prediciton
	pred_list = []
	label_list = []
	with torch.no_grad():
		for i,(imgs, labels) in enumerate(test_loader):
			# Set batch data.
			imgs, labels = imgs.to(device), labels.to(device)
			# Predict labels.
			ip1, pred = model(imgs)
			# Append predictions and labels.
			pred_list += [int(p.argmax()) for p in pred]
			label_list += [int(l) for l in labels]

	# Calculate accuracy.
	result = classification_report(pred_list, label_list)
	print(result)


def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--model_path", type=str, default='../output/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

