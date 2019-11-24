import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import random

PATH = '../models/cifar_net.pth'

def main():
	# cf. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	trainloader, testloader, classes = load_dataset()
	train(trainloader)
	test(testloader, classes)
	calc_accuracy(testloader, classes)


def load_dataset(check_data=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='../data', train=False,	download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	if check_data == True:
		check_train_data(trainloader, classes)

	return trainloader, testloader, classes


def check_train_data(trainloader, classes):
	# get some random training images
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

	# show images
	show_images(torchvision.utils.make_grid(images))
	

def show_images(img):
	# functions to show an image
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()
	img_trans = np.transpose(npimg, (1, 2, 0))
	h, w, c = img_trans.shape
	img_trans = cv2.resize(img_trans, (w*3, h*3))
	#plt.imshow(np.transpose(npimg, (1, 2, 0)))
	#plt.show()

	cv2.imshow('image', img_trans)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def train(trainloader):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = Net()
	net.to(device)
	print(device)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(2):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
				running_loss = 0.0

	print('Finished Training')

	torch.save(net.state_dict(), PATH)


def test(testloader, classes):
	dataiter = iter(testloader)
	for i in range(random.randint(0, 100)):
		images, labels = dataiter.next()

	# Ground Truth
	#show_images(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

	# Prediction
	net = Net()
	net.load_state_dict(torch.load(PATH))
	outputs = net(images)
	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
	print('')


def calc_accuracy(testloader, classes):
	net = Net()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net.to(device)
	net.load_state_dict(torch.load(PATH))
	print(device)

	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			#images, labels = data
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
	print('')

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			#images, labels = data
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i]/class_total[i]))
	print('')
	

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

if __name__ == "__main__":
	main()

