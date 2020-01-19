'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

from datasets import penn_fudan_ped
import models

import sys
sys.path.append('./vision/references/detection/')
import engine, utils


def main():
	args = parse_args()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	num_classes = 2
	train_data_loader, test_data_loader = penn_fudan_ped.get_dataset(args.data_dir)

	#model = models.get_maskrcnn_resnet50(num_classes)
	model = models.get_fasterrcnn_resnet50(num_classes)
	model.load_state_dict(torch.load(args.model_weight_path))
	model = model.to(device)
	print('Loaded a model from {}'.format(args.model_weight_path))

	engine.evaluate(model, test_data_loader, device=device)


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	# Dataset
	arg_parser.add_argument("--dataset_name", default='PennFudanPed')
	arg_parser.add_argument("--data_dir", default='D:/workspace/datasets/PennFudanPed/')

	# Model
	arg_parser.add_argument("--model_name", default='FasterRCNN-ResNet50')
	arg_parser.add_argument("--model_weight_path", default='../experiments/models/checkpoints/PennFudanPed_FasterRCNN-ResNet50_epoch=1.pth')
		
	args = arg_parser.parse_args()

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.model_weight_path)

	return args


if __name__ == "__main__":
	main()
