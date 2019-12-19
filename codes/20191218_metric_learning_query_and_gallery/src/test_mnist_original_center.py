import torch
import torch.nn as nn

import argparse
from sklearn.metrics import classification_report

from losses import CenterLoss
from mnistnet import Net
import mnist_loader

# cf. https://cpp-learning.com/center-loss/


def test(device, test_loader, model, nllloss, loss_weight, centerloss):
	model = model.eval()
		
	# Prediciton
	running_loss = 0.0
	pred_list = []
	label_list = []
	with torch.no_grad():
		for i,(imgs, labels) in enumerate(test_loader):
			# Set batch data.
			imgs, labels = imgs.to(device), labels.to(device)
			# Predict labels.
			ip1, pred = model(imgs)
			# Calculate loss.
			loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
			# Append predictions and labels.
			running_loss += loss.item()
			pred_list += [int(p.argmax()) for p in pred]
			label_list += [int(l) for l in labels]

	# Calculate accuracy.
	result = classification_report(pred_list, label_list, output_dict=True)
	test_acc = round(result['weighted avg']['f1-score'], 6)
	test_loss = round(running_loss / len(test_loader.dataset), 6)

	return test_acc, test_loss


def calc_acc_n_loss():
	args = parse_args()

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset
	train_loader, test_loader, classes = mnist_loader.load_dataset(args.dataset_dir, img_show=False)

	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	model = model.eval()
	print(model)

	# Loss
	nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
	loss_weight = 1
	centerloss = CenterLoss(10, 2).to(device)
	
	# Test a model.
	print('Testing a trained model....')
	test_acc, test_loss = test(device, test_loader, model, nllloss, loss_weight, centerloss)
	stdout_temp = 'test acc: {:<8}, test loss: {:<8}'
	print(stdout_temp.format(test_acc, test_loss))


def search_query_in_grally():
	args = parse_args()

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset
	train_loader, test_loader, classes = mnist_loader.load_dataset(args.dataset_dir, test_batch_size=4)

	# Model
	model = Net().to(device)
	model.load_state_dict(torch.load(args.model_path))
	model = model.eval()
	print(model)

	imgs, labels = iter(test_loader).next() 
	# Set batch data.
	imgs, labels = imgs.to(device), labels.to(device)
	ip1, pred = model(imgs)
	pred_list = [int(p.argmax()) for p in pred]
	label_list = [int(l) for l in labels]
	print(pred_list)
	print(label_list)



def main():
	#calc_acc_n_loss()
	search_query_in_grally()


def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='D:/workspace/datasets')
	arg_parser.add_argument("--model_path", type=str, default='../outputs/models/mnist_original_softmax_center_epoch_099.pth')
	
	args = arg_parser.parse_args()

	return args


if __name__ == "__main__":
	main()

