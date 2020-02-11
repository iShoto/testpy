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

from datasets import market1501
import metrics
import torchvision.models as models


def main():
	# Parse arguments.
	args = parse_args()

	# Load dataset.
	train_loader, gallery_loader, query_loader, class_names = market1501.load_data(args.anno_path)

	# Set device, GPU or CPU.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load a model.
	model = models.resnet50(pretrained=False)
	model.fc = nn.Linear(2048, args.n_feats)
	model.load_state_dict(torch.load(args.model_path))
	model = model.to(device)
	print(model)

	# Set a metric
	metric = metrics.ArcMarginProduct(args.n_feats, len(class_names), s=args.norm, m=args.margin, easy_margin=args.easy_margin)
	metric.load_state_dict(torch.load(args.metric_path))
	metric.to(device)
	print(metric)

	1/0


	# Query
	for i, (query_img, query_label, query_path) in enumerate(query_loader):
		with torch.no_grad():
			query_img = query_img.to(device)
			query_feat, pred = model(query_img)

	# debug
	print('Query Image Label: {}'.format(query_label.tolist()[0]))
	print('')
	
	# Gallery
	gallery_feats = []
	gallery_labels = []
	gallery_paths = []
	for i, (g_imgs, g_labels, g_paths) in enumerate(gallery_loader):
		with torch.no_grad():
			g_imgs = g_imgs.to(device)
			g_feats_temp, preds_temp = model(g_imgs)
			gallery_feats.append(g_feats_temp)
			gallery_labels.append(g_labels)
			gallery_paths += list(g_paths)  # Data type of g_paths is tuple.
	gallery_feats = torch.cat(gallery_feats, 0)
	gallery_labels = torch.cat(gallery_labels, 0)

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
	print('Search Result')
	print(df.head(20))
	print('')
	print(df['label'].value_counts())
	print('')


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

	arg_parser.add_argument('--anno_path', default='../data/annos/anno_market1501.csv')
	arg_parser.add_argument("--model_path", type=str, default='../experiments/models/model_Market1501_ResNet18_epoch=5.pth')
	arg_parser.add_argument("--metric_path", type=str, default='../experiments/models/metric_Market1501_ResNet18_epoch=5.pth')

	arg_parser.add_argument('--n_feats', default=256, type=int, help='The number of base model output')
	arg_parser.add_argument('--easy_margin', default=0, type=int, help='0 is False, 1 is True')
	arg_parser.add_argument('--weight_decay', default=0.0098, type=float, help='')
	arg_parser.add_argument('--norm', default=5, type=int, help='ArcFace: norm of input feature')
	arg_parser.add_argument('--margin', default=0.00059, type=float, help='ArcFace: margin')
	arg_parser.add_argument('--step_size', default=20, type=int, help='Learning Rate: step size')
	arg_parser.add_argument('--gamma', default=0.5, type=float, help='Learning Rate: gamma')
	
	#arg_parser.add_argument("--dataset_dir", type=str, default='../inputs/')
	#arg_parser.add_argument("--query_dir", type=str, default='../inputs/query/')
	#arg_parser.add_argument("--gallery_dir", type=str, default='../inputs/gallery/')
	#arg_parser.add_argument("--anno_path", type=str, default='../inputs/anno.csv')

	args = arg_parser.parse_args()

	# Make directory.
	#os.makedirs(args.anno_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.anno_path), args.anno_path
	assert os.path.exists(args.model_path), args.model_path
	assert os.path.exists(args.metric_path), args.metric_path


	return args


if __name__ == "__main__":
	main()

