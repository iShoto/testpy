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
	args = parse_args()
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	
	model = get_model(args.model_name)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	for epoch in range(args.n_epoch):
		train(model, device, train_loader, criterion, optimizer)
		#test(epoch)


def get_model(model_name):
	#model = VGG('VGG19')
	if model_name == 'resnet18':
		model = ResNet18()
	else:
		print('{} does NOT exist in repertory.'.format(model_name))
		sys.exit(1)
	#model = PreActResNet18()
	#model = GoogLeNet()
	#model = DenseNet121()
	#model = ResNeXt29_2x64d()
	#model = MobileNet()
	#model = MobileNetV2()
	#model = DPN92()
	#model = ShuffleNetG2()
	#model = SENet18()
	#model = ShuffleNetV2(1)
	#model = EfficientNetB0()
	
	return model
	

# Training
def train(model, device, train_loader, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
		if batch_idx % 10 == 0 and batch_idx != 0:
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

	train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

	return train_acc, train_loss


def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			if batch_idx % 50 == 0:
				print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..')
		model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, 10)
		torch.save(net.state_dict(), model_ckpt_path)
		best_acc = acc


def parse_args():
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--dataset_name", type=str, default='cifar10')
	arg_parser.add_argument("--data_dir", type=str, default='D:/workspace/datasets/')
	arg_parser.add_argument("--model_name", type=str, default='resnet18')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='../../experiments/models/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='../../experiments/models/checkpoints/{}_{}_epoch={}')
	arg_parser.add_argument('--n_epoch', default=1, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
	#arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	os.path.exists(args.data_dir)
	os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
