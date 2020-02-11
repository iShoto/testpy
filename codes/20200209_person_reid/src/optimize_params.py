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

"""
batch: 100/405, train acc: 0.03909 , train loss: 0.20076954556573737
batch: 200/405, train acc: 0.090617, train loss: 0.1915692168830046
batch: 300/405, train acc: 0.132829, train loss: 0.18499110718304138
batch: 400/405, train acc: 0.170854, train loss: 0.180213575685708
epoch:   1, train acc: 0.171718, train loss: 0.18011835154927808
batch: 100/405, train acc: 0.367335, train loss: 0.15736780662347774
batch: 200/405, train acc: 0.376965, train loss: 0.15674413837010587
batch: 300/405, train acc: 0.38767 , train loss: 0.1555511204903308
batch: 400/405, train acc: 0.391817, train loss: 0.15494201579444725
epoch:   2, train acc: 0.391102, train loss: 0.15498351782192418
batch: 100/405, train acc: 0.41835 , train loss: 0.15271492626997504
batch: 200/405, train acc: 0.417056, train loss: 0.15253686170969435
batch: 300/405, train acc: 0.415404, train loss: 0.1525446879012244
batch: 400/405, train acc: 0.414474, train loss: 0.15254244579935905
epoch:   3, train acc: 0.414696, train loss: 0.1525116750119645
batch: 100/405, train acc: 0.419596, train loss: 0.15225991164103592
batch: 200/405, train acc: 0.419364, train loss: 0.15217241481762028
batch: 300/405, train acc: 0.414419, train loss: 0.15242494464514658
batch: 400/405, train acc: 0.416057, train loss: 0.1524723521120233
epoch:   4, train acc: 0.417215, train loss: 0.152412439054913
batch: 100/405, train acc: 0.422059, train loss: 0.152291404581306
batch: 200/405, train acc: 0.420892, train loss: 0.15229961780173268
batch: 300/405, train acc: 0.418706, train loss: 0.15221507089874672
batch: 400/405, train acc: 0.418045, train loss: 0.15239195220934185
epoch:   5, train acc: 0.418132, train loss: 0.15239157187350003
[I 2020-02-11 07:20:37,909] Finished trial#71 resulted in value: 0.418132. 
 Current best value is 0.418132 with parameters: 
{'n_feats': 256, 'norm': 5, 'margin': 0.0005883992558471014, 'easy_margin': 0, 'lr': 0.08620634410578862, 'weight_decay': 0.009787166658749052}.
"""

def opt():
	study = optuna.create_study(direction='maximize')
	#study = optuna.create_study(direction='minimize')
	study.optimize(objective, n_trials=1000)

	print('Number of finished trials: ', len(study.trials))

	print('Best trial:')
	trial = study.best_trial

	print('  Value: ', trial.value)

	print('  Params: ')
	for key, value in trial.params.items():
		print('    {}: {}'.format(key, value))


def objective(trial):
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	#train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	if os.path.exists(args.anno_path) == False:
		market1501.make_csv(args.data_dir, args.anno_path)
	train_loader, test_loader, class_names = market1501.load_data(args.anno_path, args.n_batch)
		
	# Set a model.
	# cf. https://qiita.com/perrying/items/857df46bb6cdc3047bd8
	n_feats = trial.suggest_categorical('n_feats', [256*1, 256*2, 256*3, 256*4])
	model = models.resnet50(pretrained=True)
	model.fc = nn.Linear(2048, n_feats)
	model = model.to(device)
	#print(model)
	
	# Set a metric
	"""
	n_feats: 581
	norm: 1
	margin: 0.0007775271272050244
	easy_margin: 1
	"""
	norm = trial.suggest_int('norm', 0, 5)
	margin = trial.suggest_uniform('margin', 0.0, 1e-3)
	easy_margin = trial.suggest_categorical('easy_margin', [0, 1]) 
	metric = metrics.ArcMarginProduct(n_feats, len(class_names), s=norm, m=margin, easy_margin=easy_margin)
	metric.to(device)

	# Set loss function and optimization function.
	lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
	weight_decay = trial.suggest_uniform('weight_decay', 1e-6, 1e-2)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}],
						  lr=lr, 
						  weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

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
		#model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		#torch.save(model.state_dict(), model_ckpt_path)
		#print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		#print('')

	return train_acc


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
	
	#arg_parser.add_argument("--dataset_name", type=str, default='CIFAR10')
	#arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	#arg_parser.add_argument("--data_dir", type=str, default='../data/')

	arg_parser.add_argument('--dataset_name', default='Market1501')
	arg_parser.add_argument('--data_dir', default='D:/workspace/datasets/Market-1501-v15.09.15/')
	arg_parser.add_argument('--anno_dir', default='../data/annos/')
	arg_parser.add_argument('--anno_path', default='../data/annos/anno.csv')
	arg_parser.add_argument('--n_batch', default=32, type=int)

	arg_parser.add_argument("--model_name", type=str, default='ResNet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../experiments/models/checkpoints/{}_{}_epoch={}.pth')
	arg_parser.add_argument('--n_epoch', default=5, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
	arg_parser.add_argument('--n_feats', default=581, type=int, help='The number of base model output')
	arg_parser.add_argument('--easy_margin', default=1, type=int, help='0 is False, 1 is True')
	arg_parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
	arg_parser.add_argument('--norm', default=1, type=int, help='ArcFace: norm of input feature')
	arg_parser.add_argument('--margin', default=0.0008, type=float, help='ArcFace: margin')
	arg_parser.add_argument('--step_size', default=100, type=int, help='Learning Rate: step size')
	arg_parser.add_argument('--gamma', default=0.5, type=float, help='Learning Rate: gamma')

	"""
	Number of finished trials:  100
	Best trial:
	Value:  0.20689967340893214(train loss)
	Params:
		n_feats: 581
		norm: 1
		margin: 0.0007775271272050244
		easy_margin: 1
	"""
		
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
	opt()
	#main()
