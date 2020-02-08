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
import pandas as pd

#from datasets import cifar10
from datasets import market1501
import metrics
from models.resnet import ResNet18
import torchvision.models as models


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	#train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	if os.path.exists(args.anno_path) == False:
		market1501.make_csv(args.data_dir, args.anno_path)
	train_loader, test_loader, class_names = market1501.load_data(args.anno_path)
	print(len(class_names))  # 751
	
	# Set a model.
	# cf. https://qiita.com/perrying/items/857df46bb6cdc3047bd8
	model = models.resnet50(pretrained=True)
	model.fc = nn.Linear(2048, len(class_names))
	model = model.to(device)
	print(model)
	
	# cf. https://github.com/rwightman/pytorch-image-models
	# model = get_model(args.model_name, args.n_feats)
	#model = model.to(device)
	#model.load_state_dict(torch.load('../experiments/pretrained_models/resnet50_ram-a26f946b.pth'))
	#print(model)

	# Set a metric
	#metric = metrics.ArcMarginProduct(args.n_feats, len(class_names), s=args.norm, m=args.margin, easy_margin=args.easy_margin)
	#metric.to(device)

	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(device, train_loader, model, criterion, optimizer)
		#test_acc, test_loss = test(device, test_loader, model, metric, criterion)
		
		# Output score.
		#stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		#print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
		stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}'
		print(stdout_temp.format(epoch+1, train_acc, train_loss))
		
		# Save a model checkpoint.
		#model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		#torch.save(model.state_dict(), model_ckpt_path)
		#print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		#print('')


def train(device, train_loader, model, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		#print(inputs[0])
		#print(targets)
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
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
		if (batch_idx % 10 == 0 and batch_idx != 0) or (batch_idx == len(train_loader)):
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))
			
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
		features = model(inputs)
		outputs = metric_fc(features, targets)
		loss = criterion(outputs, targets)
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
		
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
	
	#arg_parser.add_argument("--dataset_name", type=str, default='CIFAR10')
	#arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	#arg_parser.add_argument("--data_dir", type=str, default='../data/')

	arg_parser.add_argument('--dataset_name', default='Market1501')
	arg_parser.add_argument('--data_dir', default='D:/workspace/datasets/Market-1501-v15.09.15/')
	arg_parser.add_argument('--anno_dir', default='../data/annos/')
	arg_parser.add_argument('--anno_path', default='../data/annos/anno.csv')

	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=100, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
	arg_parser.add_argument('--n_feats', default=512, type=int, help='The number of base model output')
	arg_parser.add_argument('--easy_margin', default=1, type=int, help='0 is False, 1 is True')
	arg_parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
	arg_parser.add_argument('--norm', default=30, type=int, help='ArcFace: norm of input feature')
	arg_parser.add_argument('--margin', default=0.5, type=float, help='ArcFace: margin')
		
	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.anno_dir, exist_ok=True)
	#os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.anno_dir)

	return args


if __name__ == "__main__":
	main()
