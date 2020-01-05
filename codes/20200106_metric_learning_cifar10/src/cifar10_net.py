import torch.nn as nn
import torch.nn.functional as F

# cf. https://cpp-learning.com/center-loss/

"""
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*5*5, 120)
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
"""

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		#self.fc3 = nn.Linear(84, 10)
		self.fc3 = nn.Linear(84, 2)
		self.fc4 = nn.Linear(2, 10)


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		ip1 = F.relu(self.fc3(x))
		ip2 = self.fc4(ip1)
		return ip1, F.log_softmax(ip2, dim=1)


class NetPre(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
		self.prelu1_1 = nn.PReLU()
		self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
		self.prelu1_2 = nn.PReLU()
		
		self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.prelu2_1 = nn.PReLU()
		self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
		self.prelu2_2 = nn.PReLU()
		
		self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
		self.prelu3_1 = nn.PReLU()
		self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
		self.prelu3_2 = nn.PReLU()
		
		self.conv4_1 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
		self.prelu4_1 = nn.PReLU()
		self.conv4_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
		self.prelu4_2 = nn.PReLU()
		
		self.ip1 = nn.Linear(256*8*8, 2)
		self.preluip1 = nn.PReLU()
		self.ip2 = nn.Linear(2, 10, bias=False)
 

	def forward(self, x):
		x = self.prelu1_1(self.conv1_1(x))
		x = self.prelu1_2(self.conv1_2(x))
		x = F.max_pool2d(x,2)
		x = self.prelu2_1(self.conv2_1(x))
		x = self.prelu2_2(self.conv2_2(x))
		x = F.max_pool2d(x,2)
		x = self.prelu3_1(self.conv3_1(x))
		x = self.prelu3_2(self.conv3_2(x))
		x = F.max_pool2d(x,2)
		x = self.prelu4_1(self.conv4_1(x))
		x = self.prelu4_2(self.conv4_2(x))
		x = F.max_pool2d(x,2)
		x = x.view(-1, 256*8*8)
		ip1 = self.preluip1(self.ip1(x))
		ip2 = self.ip2(ip1)
		return ip1, F.log_softmax(ip2, dim=1)
