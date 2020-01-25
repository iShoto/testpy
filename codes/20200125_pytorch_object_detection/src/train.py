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
	
	if not os.path.exists(args.anno_path):
		penn_fudan_ped.make_csv(args.data_dir, args.anno_path)
	train_data_loader, test_data_loader = penn_fudan_ped.get_dataset(args.anno_path)

	print('Loading a model...')
	num_classes = 2
	model = models.get_fasterrcnn_resnet50(num_classes)
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

	# let's train it for 10 epochs
	for epoch in range(args.n_epoch):
		# train for one epoch, printing every 10 iterations
		engine.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
		# update the learning rate
		lr_scheduler.step()
		# evaluate on the test dataset
		engine.evaluate(model, test_data_loader, device=device)

		# Save a model checkpoint.
		model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		torch.save(model.state_dict(), model_ckpt_path)
		print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		print('')


#def train(model, optimizer, train_data_loader, device, print_freq=10):
#	model.train()


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser()
	
	# Dataset
	arg_parser.add_argument("--dataset_name", default='PennFudanPed')
	arg_parser.add_argument("--data_dir", default='D:/workspace/datasets/PennFudanPed/')
	arg_parser.add_argument('--anno_dir', default='../data/annos/')
	arg_parser.add_argument('--anno_path', default='../data/annos/anno_penn-fudan-ped.csv')

	# Model
	arg_parser.add_argument("--model_name", default='FasterRCNN-ResNet50')
	arg_parser.add_argument("--model_ckpt_dir", default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')

	# Others
	arg_parser.add_argument("--n_epoch", default=10, type=int)
		
	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.anno_dir, exist_ok=True)
	os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.anno_dir)
	assert os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
