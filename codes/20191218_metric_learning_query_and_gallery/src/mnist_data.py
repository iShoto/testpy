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


def load_dataset(dataset_dir, train_batch_size=128, test_batch_size=128, img_show=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
	train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	if img_show == True:
		show_data(train_loader)

	return train_loader, test_loader, classes


def show_data(data_loader):
	images, labels = iter(data_loader).next()  # data_loader のミニバッチの image を取得
	img = torchvision.utils.make_grid(images, nrow=16, padding=1)  # nrom*nrom のタイル形状の画像を作る
	plt.ion()
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 画像を matplotlib 用に変換
	plt.draw()
	plt.pause(3)  # Display an image for three seconds.
	plt.close()


"""
Object ReID
"""
def make_query_and_gallery(dataset_dir, query_dir, gallery_dir):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	q_idx = random.choice(range(len(testset)))
	g_idxs= random.sample(range(len(testset)), 100)
	q_img, q_label = testset[q_idx]

	# Save query image.
	if os.path.exists(query_dir) == True:
		shutil.rmtree(query_dir)
	os.makedirs(query_dir)
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
		if self.transform:
			image = self.transform(image)
		
		return image, label


def load_query_and_gallery(anno_path, img_show=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	# Query
	query_dataset = ReIDDataset(anno_path, 'query', transform)
	query_loader = DataLoader(query_dataset, batch_size=len(query_dataset), shuffle=False)
	#for i,(imgs, labels) in enumerate(query_loader):
	#	print(imgs.shape)
	#	print(labels)
	#print('')
	
	# Gallery
	gallery_dataset = ReIDDataset(anno_path, 'gallery', transform)
	gallery_loader = DataLoader(gallery_dataset, batch_size=len(gallery_dataset), shuffle=True)
	#for i,(imgs, labels) in enumerate(gallery_loader):
	#	print(imgs.shape)
	#	print(labels)
	#print('')

	# Class
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	# debug
	print('num query: {}, num gallery: {}'.format(len(query_dataset), len(gallery_dataset)))
	if img_show == True:
		show_data(gallery_loader)

	return query_loader, gallery_loader, classes
