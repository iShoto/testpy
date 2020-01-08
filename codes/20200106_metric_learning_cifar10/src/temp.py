from __future__ import print_function

import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import argparse
from tqdm import trange

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
import torchvision

from config import config
from utils import visualizer, view_model
from data import dataset
from models import resnet, metrics
#from test import *
#import test
from losses import CenterLoss
from cifar10_net import Net
import cifar10_data


def main():
	args = parse_args()

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset
	train_loader, test_loader, classes = cifar10_data.load_dataset(args.dataset_dir, img_show=True)

	model = resnet.resnet18()
	print(model)

	num_classes = 10
	easy_margin = False
	metric_fc = metrics.ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=easy_margin)
	
	model.to(device)
	model = DataParallel(model)
	metric_fc.to(device)
	metric_fc = DataParallel(metric_fc)

	lr = 1e-1  # initial learning rate
	lr_step = 10
	weight_decay = 5e-4
	optimizer = torch.optim.SGD([{'params': model.parameters()}, 
								 {'params': metric_fc.parameters()}], 
								 lr=lr, weight_decay=weight_decay)
	scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)

	max_epoch = 2
	for i in range(max_epoch):
		scheduler.step()

		model.train()
		for ii,(imgs, labels) in enumerate(train_loader):
			# Set batch data.
			imgs, labels = imgs.to(device), labels.to(device).long()
			feature = model(imgs)
			output = metric_fc(feature, labels)
			loss = criterion(output, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print(loss)


def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--model_path_temp", type=str, default='../outputs/models/checkpoints/mnist_original_softmax_center_epoch_{}.pth')
	arg_parser.add_argument("--vis_img_path_temp", type=str, default='../outputs/visual/epoch_{}.png')
	
	args = arg_parser.parse_args()

	return args


if __name__ == '__main__':
	main()