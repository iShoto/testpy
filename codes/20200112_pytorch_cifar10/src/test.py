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

from models import *
from datasets import cifar10


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	
	# Load a model.
	model = get_model(args.model_name)
	model.load_state_dict(torch.load(args.model_path))
	model = model.to(device)
	print('Loaded a model from {}'.format(args.model_path))

	# Define loss function.
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


def get_model(model_name):
	if model_name == 'VGG19':
		model = VGG('VGG19')
	elif model_name == 'ResNet18':
		model = ResNet18()
	elif model_name == 'PreActResNet18':
		model = PreActResNet18()
	elif model_name == 'GoogLeNet':
		model = GoogLeNet()
	elif model_name == 'DenseNet121':
		model = DenseNet121()
	elif model_name == 'ResNeXt29_2x64d':
		model = ResNeXt29_2x64d()
	elif model_name == 'MobileNet':
		model = MobileNet()
	elif model_name == 'MobileNetV2':
		model = MobileNetV2()
	elif model_name == 'DPN92':
		model = DPN92()
	elif model_name == 'ShuffleNetG2':
		model = ShuffleNetG2()
	elif model_name == 'SENet18':
		model = SENet18()
	elif model_name == 'ShuffleNetV2':
		model = ShuffleNetV2(1)
	elif model_name == 'EfficientNetB0':
		model = EfficientNetB0()
	else:
		print('{} does NOT exist in repertory.'.format(model_name))
		sys.exit(1)
	
	return model


def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_path", type=str, default='../experiments/models/CIFAR10_ResNet18_epoch=10.pth')

	args = arg_parser.parse_args()

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.model_path)

	return args


if __name__ == "__main__":
	main()
