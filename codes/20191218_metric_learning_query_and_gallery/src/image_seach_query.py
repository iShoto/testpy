import torch
import torch.nn as nn
from torchvision import datasets, transforms

import os
import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2
import random
import scipy.misc
import shutil

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


class MNISTDataset(Dataset):
	def __init__(self, csv_file_path, root_dir, transform=None):
		#pandasでcsvデータの読み出し
		self.image_dataframe = pd.read_csv(csv_file_path)
		self.root_dir = root_dir
		#画像データへの処理
		self.transform = transform

	def __len__(self):
		return len(self.image_dataframe)

	def __getitem__(self, idx):
		#dataframeから画像へのパスとラベルを読み出す
		label = self.image_dataframe.iat[idx, LABEL_IDX]
		img_name = os.path.join(self.root_dir, 'classification-of-handwritten-letters',
				'letters2', self.image_dataframe.iat[idx, IMG_IDX])
		#画像の読み込み
		image = io.imread(img_name)
		#画像へ処理を加える
		if self.transform:
			image = self.transform(image)

		return image, label


def main():
	"""
	set query and gallery.
	get gallery features.
	get query feature.
	"""
	args = parse_args()
	#make_query_and_gallery(args.dataset_dir, args.query_dir, args.gallery_dir)

	mnist_dataset = datasets.ImageFolder(root='../inputs/')#, transform=data_transform)
	print(mnist_dataset)
	
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



def make_query_and_gallery(dataset_dir, query_dir, gallery_dir):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	q_idx = random.choice(range(len(testset)))
	g_idxs= random.sample(range(len(testset)), 100)
	q_img, q_label = testset[q_idx]

	# Save query image.
	if os.path.exists(query_dir) == True:
		shutil.rmtree(query_dir)
	os.makedirs(query_dir)
	scipy.misc.imsave(query_dir+'{}_{}.png'.format(q_label, q_idx), q_img.numpy()[0])
	
	# Save gallery images.
	if os.path.exists(gallery_dir) == True:
		shutil.rmtree(gallery_dir)
	os.makedirs(gallery_dir)
	for g_idx in g_idxs:
		g_img, g_label = testset[g_idx]
		scipy.misc.imsave(gallery_dir+'{}_{}.png'.format(g_label, g_idx), g_img.numpy()[0])


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
	arg_parser.add_argument("--query_dir", type=str, default='../inputs/query/')
	arg_parser.add_argument("--gallery_dir", type=str, default='../inputs/gallery/')

	arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

