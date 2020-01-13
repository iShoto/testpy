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

from datasets import cifar10
import metrics
#from models import *
from models.resnet_ex import ResNet18


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	
	# Set a model.
	model = get_model(args.model_name, args.n_feats)
	model = model.to(device)
	print(model)

	#easy_margin = False
	metric_fc = metrics.ArcMarginProduct(args.n_feats, len(class_names), s=30, m=0.5, easy_margin=False)
	metric_fc.to(device)

	# Set loss function and optimization function.
	#criterion = nn.CrossEntropyLoss()
	#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
						  lr=args.lr, 
						  weight_decay=args.weight_decay)

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(device, train_loader, model, metric_fc, criterion, optimizer)
		test_acc, test_loss = test(device, test_loader, model, metric_fc, criterion)
		
		# Output score.
		stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))

		# Save a model checkpoint.
		model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		torch.save(model.state_dict(), model_ckpt_path)
		print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		print('')


def train(device, train_loader, model, metric_fc, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device).long()
		print(inputs.shape)
		features = model(inputs)
		outputs = metric_fc(features, targets)
		loss = criterion(outputs, targets)
		#print(features)
		#print(outputs)
		
		if batch_idx > 1:
			1/0

		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()

		# Calculate score at present.
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
		if batch_idx % 10 == 0 and batch_idx != 0:
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))
			1/0


	# Calculate score.
	train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

	return train_acc, train_loss


def test(device, test_loader, model, metric_fc, criterion):
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
		
	test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

	return test_acc, test_loss


def get_model(model_name, n_feats):
	if model_name == 'VGG19':
		model = VGG('VGG19')
	elif model_name == 'ResNet18':
		model = ResNet18(n_feats)
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
	
	arg_parser.add_argument("--dataset_name", type=str, default='CIFAR10')
	arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	#arg_parser.add_argument("--data_dir", type=str, default='../data/')
	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=1, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
	arg_parser.add_argument('--n_feats', default=1, type=int, help='The number of base model output')
	arg_parser.add_argument('--easy_margin', default=0, type=int, help='0 is False, 1 is True')
	arg_parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
	#easy_margin = False

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.data_dir, exist_ok=True)
	os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
