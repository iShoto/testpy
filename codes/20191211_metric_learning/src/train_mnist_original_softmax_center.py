import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import Function

import matplotlib.pyplot as plt
import argparse
from tqdm import trange

from losses import CenterLoss
from mnistnet import Net

# cf. https://cpp-learning.com/center-loss/


def load_dataset(dataset_dir):
	# Dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
	train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

	return train_loader


def train(train_loader, device, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer):
	ip1_loader = []
	idx_loader = []
	for i,(data, labels) in enumerate(train_loader):
		# Set batch data.
		data, labels = data.to(device), labels.to(device)
		# Predict labels.
		ip1, pred = model(data)
		# Calculate loss.
		loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
		# Initilize gradient.
		dnn_optimizer.zero_grad()
		center_optimizer.zero_grad()
		# Calculate gradient.
		loss.backward()
		# Upate parameters.
		dnn_optimizer.step()
		center_optimizer.step()

		ip1_loader.append(ip1)
		idx_loader.append((labels))
 
	feat = torch.cat(ip1_loader, 0)
	labels = torch.cat(idx_loader, 0)

	return feat, labels


def visualize(feat, labels, epoch, vis_img_path):
	#plt.ion()
	colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
			  '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	plt.clf()
	for i in range(10):
		plt.plot(feat[labels==i, 0], feat[labels==i, 1], '.', color=colors[i])
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='best')
	plt.xlim(left=-8, right=8)
	plt.ylim(bottom=-8, top=8)
	plt.text(-7.8, 7.3, "epoch=%d" % epoch)
	plt.savefig(vis_img_path)


def main():
	args = parse_args()

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset
	train_loader = load_dataset(args.dataset_dir)

	# Model
	model = Net().to(device)
	print(model)

	# Loss
	nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
	loss_weight = 1
	centerloss = CenterLoss(10, 2).to(device)
	
	# Optimizer
	dnn_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
	sheduler = lr_scheduler.StepLR(dnn_optimizer, 20, gamma=0.8)
	center_optimizer = optim.SGD(centerloss.parameters(), lr =0.5)
	
	#for epoch in range(100):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	for epoch in trange(2, desc='Training The Model'):
		sheduler.step()
		feat, labels = train(train_loader, device, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer)
		vis_img_path = args.vis_img_path_temp.format(str(epoch+1).zfill(3))
		visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, vis_img_path)

		model_name_path = args.model_name_path_temp.format(str(epoch).zfill(3))
		torch.save(model.state_dict(), model_name_path)


def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--model_name_path_temp", type=str, default='../output/models/mnist_original_softmax_center_epoch_{}.pth')
	arg_parser.add_argument("--vis_img_path_temp", type=str, default='../output/visual/epoch_{}.png')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

