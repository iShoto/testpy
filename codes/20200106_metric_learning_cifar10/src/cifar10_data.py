import torch
from torchvision import datasets, transforms
from  torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import io

import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import pandas as pd
import cv2


def load_dataset(dataset_dir, train_batch_size=16, valid_batch_size=16, img_show=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	
	train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)
	valid_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,	download=True, transform=transform)
	valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=valid_batch_size, shuffle=False, num_workers=0)
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	if img_show == True:
		show_data(train_loader, classes)

	return train_loader, valid_loader, classes


def show_data(data_loader, classes):
	# Get images and labels.
	images, labels = iter(data_loader).next()
	print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))
		
	# Transfrom an image.
	img = torchvision.utils.make_grid(images)
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()
	img_trans = np.transpose(npimg, (1,2,0))
	h, w, c = img_trans.shape
	img_trans = cv2.resize(img_trans, (w*3, h*3))

	# Show an image.
	cv2.imshow('image', img_trans)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


"""
Object ReID
"""
def make_query_and_gallery(dataset_dir, query_dir, gallery_dir):
	# 
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	q_idx = random.choice(range(len(testset)))
	g_idxs= random.sample(range(len(testset)), 100)
	
	# Save query image.
	if os.path.exists(query_dir) == True:
		shutil.rmtree(query_dir)
	os.makedirs(query_dir)
	q_img, q_label = testset[q_idx]
	scipy.misc.imsave(query_dir+'{}_{}.png'.format(q_label, q_idx), q_img.numpy()[0])
	
	# Save gallery images.
	if os.path.exists(gallery_dir) == True:
		shutil.rmtree(gallery_dir)
	os.makedirs(gallery_dir)
	for g_idx in g_idxs:
		g_img, g_label = testset[g_idx]
		scipy.misc.imsave(gallery_dir+'{}_{}.png'.format(g_label, g_idx), g_img.numpy()[0])


def make_anno_file(query_dir, gallery_dir, anno_path):
	annos = []
	annos += __set_annos(query_dir, 'query')
	annos += __set_annos(gallery_dir, 'gallery')
	df = pd.DataFrame(annos)
	df.to_csv(anno_path, index=False)
	#print(df)


def __set_annos(img_dir, data_type):
	annos = []
	for d in os.listdir(img_dir):
		dic = {}
		dic['data_type'] = data_type
		dic['img_name'] = d
		dic['img_path'] = img_dir + d
		dic['label'] = d.split('_')[0]
		dic['id'] = d.split('.')[0].split('_')[1]
		annos.append(dic)

	return annos


class ReIDDataset(Dataset):
	def __init__(self, anno_path, data_type, transform=None):
		df_all = pd.read_csv(anno_path)
		self.df = df_all[df_all['data_type']==data_type].reset_index(drop=True)  # Filter data by query or gallery.
		self.transform = transform


	def __len__(self):
		return len(self.df)


	def __getitem__(self, idx):
		img_path = self.df.loc[idx, 'img_path']
		assert os.path.exists(img_path)
		image = io.imread(img_path)
		label = self.df.loc[idx, 'label']
		img_path = self.df.loc[idx, 'img_path']
		if self.transform:
			image = self.transform(image)
		
		return image, label, img_path


def load_query_and_gallery(anno_path, img_show=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	# Query
	query_dataset = ReIDDataset(anno_path, 'query', transform)
	query_loader = DataLoader(query_dataset, batch_size=len(query_dataset), shuffle=False)
		
	# Gallery
	gallery_dataset = ReIDDataset(anno_path, 'gallery', transform)
	#gallery_loader = DataLoader(gallery_dataset, batch_size=len(gallery_dataset), shuffle=True)
	gallery_loader = DataLoader(gallery_dataset, batch_size=8, shuffle=True)
	
	# Class
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	# debug
	print('num query: {}, num gallery: {}'.format(len(query_dataset), len(gallery_dataset)))
	print('')
	if img_show == True:
		show_data(gallery_loader)

	return query_loader, gallery_loader, classes
