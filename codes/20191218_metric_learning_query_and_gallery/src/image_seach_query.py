import torch
import torch.nn as nn
from torchvision import datasets, transforms

import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2

from losses import CenterLoss
from mnist_net import Net
import mnist_loader

"""
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(dataset_dir, train_batch_size=128, test_batch_size=128, img_show=False):
	# Dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
"""


def main():
	"""
	set query and gallery.
	get gallery features.
	get query feature.
	"""
	args = parse_args()

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)
	x, _ = testset[7777]
	print(x.numpy().shape)
	print(_)
	import scipy.misc
	scipy.misc.imsave('outfile.png', x.numpy()[0])
	1/0
	cv2.imshow(x.numpy())
	#plt.imshow(x.numpy()[0], cmap='gray')
	#plt.show()

	1/0


	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset
	train_loader, test_loader, classes = mnist_loader.load_dataset(args.dataset_dir, img_show=False)

	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	print(model)

	for i,(imgs, labels) in enumerate(test_loader):
		with torch.no_grad():
			# Set batch data.
			imgs, labels = imgs.to(device), labels.to(device)
			# Predict labels.
			ip1, pred = model(imgs)
			# Calculate loss.
			loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
			# Append predictions and labels.
			running_loss += loss.item()
			pred_list += [int(p.argmax()) for p in pred]
			label_list += [int(l) for l in labels]


def test(device, test_loader, model, nllloss, loss_weight, centerloss):
	model = model.eval()
		
	# Prediciton
	running_loss = 0.0
	pred_list = []
	label_list = []
	with torch.no_grad():
		for i,(imgs, labels) in enumerate(test_loader):
			# Set batch data.
			imgs, labels = imgs.to(device), labels.to(device)
			# Predict labels.
			ip1, pred = model(imgs)
			# Calculate loss.
			loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
			# Append predictions and labels.
			running_loss += loss.item()
			pred_list += [int(p.argmax()) for p in pred]
			label_list += [int(l) for l in labels]

	# Calculate accuracy.
	result = classification_report(pred_list, label_list, output_dict=True)
	test_acc = round(result['weighted avg']['f1-score'], 6)
	test_loss = round(running_loss / len(test_loader.dataset), 6)

	return test_acc, test_loss

def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

