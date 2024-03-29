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
import optuna

from datasets import market1501
import metrics
import torchvision.models as models


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	if os.path.exists(args.anno_path) == False:
		market1501.make_train_anno(args.data_dir, args.anno_path)
	train_loader, gallery_loader, query_loader, class_names = market1501.load_train_data(args.anno_path, args.n_batch)
		
	# Set a model.
	# cf. https://qiita.com/perrying/items/857df46bb6cdc3047bd8
	model = models.resnet50(pretrained=True)
	model.fc = nn.Linear(2048, args.n_feats)
	model = model.to(device)
	print(model)
	
	# Set a metric
	metric = metrics.ArcMarginProduct(args.n_feats, len(class_names), s=args.norm, m=args.margin, easy_margin=args.easy_margin)
	metric.to(device)

	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}],
						  lr=args.lr, 
						  weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(device, train_loader, args.n_batch, model, metric, criterion, optimizer, scheduler)
		#test_acc, test_loss = test(device, test_loader, model, metric, criterion)
		
		# Output score.
		#stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		#print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
		stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}'
		print(stdout_temp.format(epoch+1, train_acc, train_loss))

		# Save a model checkpoint.
		model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		metric_ckpt_path = args.metric_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		torch.save(model.state_dict(), model_ckpt_path)
		torch.save(metric.state_dict(), metric_ckpt_path)
		print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		print('Saved a metric checkpoint at {}'.format(metric_ckpt_path))
		print('')


def train(device, train_loader, n_batch, model, metric_fc, criterion, optimizer, scheduler):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device).long()
		features = model(inputs)
		outputs = metric_fc(features, targets)
		loss = criterion(outputs, targets)
		
		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step() 
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
		
		# Calculate score at present.
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, n_batch, batch_idx, train_loader)
		if (batch_idx % 100 == 0 and batch_idx != 0) or (batch_idx == len(train_loader)):
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))
			
	# Calculate score.
	#train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

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


def calc_score(output_list, target_list, running_loss, n_batch, batch_idx, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	#loss = round(running_loss / len(data_loader.dataset), 6)
	if n_batch * batch_idx < len(data_loader.dataset):
		loss = running_loss / (n_batch * (batch_idx+1))
	else:
		loss = running_loss / len(data_loader.dataset)
	
	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument('--dataset_name', default='Market1501')
	arg_parser.add_argument('--data_dir', default='D:/workspace/datasets/Market-1501-v15.09.15/')
	arg_parser.add_argument('--anno_dir', default='../data/annos/')
	arg_parser.add_argument('--anno_path', default='../data/annos/anno_market1501_train.csv')
	arg_parser.add_argument('--n_batch', default=32, type=int)

	arg_parser.add_argument("--model_name", type=str, default='ResNet50')
	arg_parser.add_argument("--ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/model_{}_{}_epoch={}.pth')
	arg_parser.add_argument("--metric_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/metric_{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=5, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.086, type=float, help='Learning rate')
	arg_parser.add_argument('--n_feats', default=256, type=int, help='The number of base model output')
	arg_parser.add_argument('--easy_margin', default=0, type=int, help='0 is False, 1 is True')
	arg_parser.add_argument('--weight_decay', default=0.0098, type=float, help='')
	arg_parser.add_argument('--norm', default=5, type=int, help='ArcFace: norm of input feature')
	arg_parser.add_argument('--margin', default=0.00059, type=float, help='ArcFace: margin')
	arg_parser.add_argument('--step_size', default=20, type=int, help='Learning Rate: step size')
	arg_parser.add_argument('--gamma', default=0.5, type=float, help='Learning Rate: gamma')

	"""
	{'n_feats': 256, 'norm': 5, 'margin': 0.0005883992558471014, 'easy_margin': 0, 'lr': 0.08620634410578862, 'weight_decay': 0.009787166658749052}.
	"""

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.anno_dir, exist_ok=True)
	os.makedirs(args.ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.anno_dir)
	assert os.path.exists(args.ckpt_dir)

	return args


if __name__ == "__main__":
	main()
