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
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 1
	for epoch in range(num_epochs):
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


def train(model, optimizer, train_data_loader, device, print_freq=10):
	model.train()


	"""
	header = 'Epoch: [{}]'.format(epoch)

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1. / 1000
		warmup_iters = min(1000, len(data_loader) - 1)

		lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

	for images, targets in metric_logger.log_every(data_loader, print_freq, header):
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		loss_dict = model(images, targets)

		losses = sum(loss for loss in loss_dict.values())

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		losses_reduced = sum(loss for loss in loss_dict_reduced.values())

		loss_value = losses_reduced.item()

		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			print(loss_dict_reduced)
			sys.exit(1)

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()

		if lr_scheduler is not None:
			lr_scheduler.step()

		metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])
	"""


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	# Dataset
	arg_parser.add_argument("--dataset_name", default='PennFudanPed')
	arg_parser.add_argument("--data_dir", default='D:/workspace/datasets/PennFudanPed/')

	# Model
	arg_parser.add_argument("--model_name", default='FasterRCNN-ResNet50')
	arg_parser.add_argument("--model_ckpt_dir", default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')
		
	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
