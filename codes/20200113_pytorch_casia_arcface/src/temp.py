from __future__ import print_function

import os
import numpy as np
import random
import time

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
import test


def main():
	device = torch.device("cuda")

	model = resnet.resnet18()
	print(model)

	num_classes = 13938
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



if __name__ == '__main__':
	main()