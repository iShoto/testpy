import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io
from PIL import Image

import os
import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2
import random
import scipy.misc
import shutil
import pandas as pd
import numpy as np

from losses import CenterLoss
from mnist_net import Net
import mnist_data

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
	#make_query_and_gallery_from_mnist(args.dataset_dir, args.query_dir, args.gallery_dir, args.anno_path)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	query_loader, gallery_loader, classes = mnist_data.load_query_and_gallery(args.anno_path, img_show=False)

	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()

	# Query
	img_path = args.query_dir + os.listdir(args.query_dir)[0]
	query_img = Image.open(img_path)
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	query_img = torch.unsqueeze(transform(query_img), 0)
	query_img = query_img.to(device)
	with torch.no_grad():
		query_img = query_img.to(device)
		query_feat, pred = model(query_img)
	
	# Gallery
	for i, (gallery_imgs, gallery_labels, gallery_paths) in enumerate(gallery_loader):
		with torch.no_grad():
			gallery_imgs = gallery_imgs.to(device)
			gallery_feats, pred = model(gallery_imgs)

	# Calculate cosine similarity.
	dist_mat = cosine_similarity(query_feat, gallery_feats)

	# Organize ReID ranking.
	lis = []
	for i in range(len(gallery_paths)):
		dic = {}
		dic['dist'] = dist_mat.tolist()[0][i]
		dic['label'] = np.array(gallery_labels).tolist()[i]
		dic['img_path'] = gallery_paths[i]
		lis.append(dic)
	df = pd.DataFrame(lis)
	df = df.sort_values(by=['dist'], ascending=True)
	df = df.reset_index(drop=True)
	print(df)


def cosine_similarity(qf, gf):
	epsilon = 0.00001
	dist_mat = qf.mm(gf.t())
	qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True) #mx1
	gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True) #nx1
	qg_normdot = qf_norm.mm(gf_norm.t())

	dist_mat = dist_mat.mul(1/qg_normdot).cpu().numpy()
	dist_mat = np.clip(dist_mat, -1+epsilon,1-epsilon)
	dist_mat = np.arccos(dist_mat)
	return dist_mat



def make_query_and_gallery_from_mnist(dataset_dir, query_dir, gallery_dir, anno_path):
	mnist_data.make_query_and_gallery(args.dataset_dir, args.query_dir, args.gallery_dir)
	mnist_data.make_anno_file(args.query_dir, args.gallery_dir, args.anno_path)


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
	arg_parser.add_argument("--input_dir", type=str, default='../inputs/')
	arg_parser.add_argument("--query_dir", type=str, default='../inputs/query/')
	arg_parser.add_argument("--gallery_dir", type=str, default='../inputs/gallery/')
	arg_parser.add_argument("--anno_path", type=str, default='../inputs/anno.csv')


	arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

