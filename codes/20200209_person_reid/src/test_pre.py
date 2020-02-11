'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from sklearn.metrics import classification_report

from datasets import market1501
import metrics
import torchvision.models as models


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	train_loader, gallery_loader, query_loader, class_names = market1501.load_data(args.anno_path)
	
	# Load a model.
	model = models.resnet50(pretrained=True)
	model.fc = nn.Linear(2048, args.n_feats)
	model.load_state_dict(torch.load(args.model_path))
	model = model.to(device)
	print(model)
	1/0

	# Set a metric
	metric = metrics.ArcMarginProduct(args.n_feats, len(class_names), s=args.norm, m=args.margin, easy_margin=args.easy_margin)
	metric.to(device)

	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	
	# Test a model.
	test_acc, test_loss = test(model, device, test_loader, criterion)

	# Output score.
	stdout_temp = 'test acc: {:<8}, test loss: {:<8}'
	print(stdout_temp.format(test_acc, test_loss))


def test(model, device, test_loader, criterion):
	model.eval()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
	
	# Calculate score.
	test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

	return test_acc, test_loss


def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument('--anno_path', type=str, default='../../data/annos/anno_market1501.csv')
	arg_parser.add_argument('--model_path', type=str, default='../experiments/models/Market1501_ResNet18_epoch=83.pth')

	#arg_parser.add_argument('--data_dir', type=str, default='../data/')
	#arg_parser.add_argument('--model_name', type=str, default='ResNet18')
	#arg_parser.add_argument('--model_path', type=str, default='../experiments/models/CIFAR10_ResNet18_epoch=10.pth')

	args = arg_parser.parse_args()

	# Validate paths.
	assert os.path.exists(args.anno_path)
	assert os.path.exists(args.model_path)

	return args


if __name__ == "__main__":
	main()
