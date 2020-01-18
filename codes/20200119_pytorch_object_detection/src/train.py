'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import argparse

from datasets.penn_fudan import PennFudanDataset

import vision.references.detection.transforms as T
from vision.references.detection import engine, utils


def main():
	# train on the GPU or on the CPU, if a GPU is not available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	# our dataset has two classes only - background and person
	num_classes = 2
	# use our dataset and defined transformations
	dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
	dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

	# get the model using our helper function
	model = get_model_instance_segmentation(num_classes)

	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
								momentum=0.9, weight_decay=0.0005)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 10

	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
		# update the learning rate
		lr_scheduler.step()
		# evaluate on the test dataset
		engine.evaluate(model, data_loader_test, device=device)

	print("That's it!")


def get_model(num_classes):
	# load an instance segmentation model pre-trained pre-trained on COCO
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	# now get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	# and replace the mask predictor with a new one
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
													   hidden_layer,
													   num_classes)

	return model


def get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)
	

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--dataset_name", default='CIFAR10')
		
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()
