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


def main():
	# Parse arguments.
	args = parse_args()

	# Get dataset.
	make_query_and_gallery_from_mnist(args.dataset_dir, args.query_dir, args.gallery_dir, args.anno_path)
	query_loader, gallery_loader, classes = mnist_data.load_query_and_gallery(args.anno_path, img_show=False)

	# Set device, GPU or CPU.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	model.eval()

	# Query
	for i, (query_img, query_label, query_path) in enumerate(query_loader):
		with torch.no_grad():
			query_img = query_img.to(device)
			query_feat, pred = model(query_img)
		
	# Gallery
	for i, (gallery_imgs, gallery_labels, gallery_paths) in enumerate(gallery_loader):
		with torch.no_grad():
			gallery_imgs = gallery_imgs.to(device)
			gallery_feats, pred = model(gallery_imgs)

	# Calculate cosine similarity.
	dist_matrix = cosine_similarity(query_feat, gallery_feats)

	# Organize ReID ranking.
	lis = []
	for i in range(len(gallery_paths)):
		dic = {}
		dic['dist'] = dist_matrix.tolist()[0][i]
		dic['label'] = np.array(gallery_labels).tolist()[i]
		dic['img_path'] = gallery_paths[i]
		lis.append(dic)
	df = pd.DataFrame(lis)
	df = df.sort_values(by=['dist'], ascending=True)
	df = df.reset_index(drop=True)

	# debug
	print(df)
	print(df['label'].value_counts())


def make_query_and_gallery_from_mnist(dataset_dir, query_dir, gallery_dir, anno_path):
	mnist_data.make_query_and_gallery(dataset_dir, query_dir, gallery_dir)
	mnist_data.make_anno_file(query_dir, gallery_dir, anno_path)


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


def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--query_dir", type=str, default='../inputs/query/')
	arg_parser.add_argument("--gallery_dir", type=str, default='../inputs/gallery/')
	arg_parser.add_argument("--anno_path", type=str, default='../inputs/anno.csv')
	arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

