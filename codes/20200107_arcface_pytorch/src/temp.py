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
from models import resnet
#from test import *
import test


def main():
	model = resnet.resnet18()
	print(model)



if __name__ == '__main__':
	main()