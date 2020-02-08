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
from tqdm import trange
import pandas as pd
from PIL import Image


class Market1501(object):
	def __init__(self, anno_path, mode, transforms=None):
		df_all = pd.read_csv(anno_path)
		self.df = df_all[df_all['mode']==mode].reset_index(drop=True)
		self.transforms = transforms

	def __getitem__(self, idx):
		# Filter data
		df = self.df.copy()
		
		# Image
		img_path = df.loc[idx, 'image_path']
		assert os.path.exists(img_path)
		img = Image.open(img_path).convert('RGB')
		img = img.resize([img.size[0], img.size[0]], Image.NEAREST)

		# Target
		#person_id = df.loc[idx, 'person_id']
		person_index = df.loc[idx, 'person_index']

		# Transform
		if self.transforms is not None:
			img = self.transforms(img)

		return img, person_index

	def __len__(self):
		return len(set(self.df['image_path'].values.tolist()))


def load_data(anno_path, n_batch=32):
	transform_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	train_set = Market1501(anno_path, 'train', transforms=transform_train)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=n_batch, shuffle=True, num_workers=0)
	test_set = Market1501(anno_path, 'test', transforms=transform_train)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=n_batch, shuffle=False, num_workers=0)
	df = pd.read_csv(anno_path)
	#class_names = sorted(list(set(df.loc[df['mode']=='train', 'person_id'].values)))
	class_names = sorted(list(set(df.loc[df['mode']=='train', 'person_index'].values)))
		
	return train_loader, test_loader, class_names


def make_csv(data_dir, anno_path):
	#args = parse_args()
	
	# Set data directories.
	train_dir = data_dir + 'bounding_box_train/'
	test_dir = data_dir + 'bounding_box_test/'

	# Get image names.
	train_img_names = sorted([d for d in os.listdir(train_dir) if d.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')])
	test_img_names = sorted([d for d in os.listdir(test_dir) if d.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')])
	
	# Organize anntation data.
	train_list = __org_img_data(train_dir, train_img_names, 'train')
	test_list = __org_img_data(test_dir, test_img_names, 'test')

	# Make and save DataFrame.
	__make_and_save_dataframe(train_list, test_list, anno_path)


def __org_img_data(data_dir, img_names, mode):
	lis = []
	for i in trange(len(img_names), desc='Organizing {} data'.format(mode)):
		dic = {}
		dic['image_name'] = img_names[i]
		dic['image_path'] = data_dir + img_names[i]
		splited = img_names[i].split('_')
		dic['person_id'] = splited[0]
		dic['camera_id'] = splited[1][:2]
		dic['sequence_id'] = splited[1][2:]
		dic['frame_no'] = splited[2]
		dic['dpm_bbox_no'] = splited[3].split('.')[0]  # DPM: Deformable Part Model, bbox: bounding box
		dic['mode'] = mode
		lis.append(dic)

	return lis


def __make_and_save_dataframe(train_list, test_list, anno_path):
	# Make DataFrame.
	df = pd.DataFrame(train_list+test_list)

	# Get person IDs.
	train_person_ids = sorted(set(df.loc[df['mode']=='train', 'person_id'].values))
	test_person_ids = sorted(set(df.loc[df['mode']=='test', 'person_id'].values))
	test_person_ids.remove('-1')
	test_person_ids.remove('0000')

	# Make person ID and index dictonary.
	person_dic = {}
	train_person_dic = {train_person_ids[i]: i for i in range(len(train_person_ids))}
	test_person_dic = {test_person_ids[i]: i+len(train_person_ids) for i in range(len(test_person_ids))}
	test_person_dic_neg = {'-1': 1501, '0000': 1502}
	person_dic.update(train_person_dic)
	person_dic.update(test_person_dic)
	person_dic.update(test_person_dic_neg)
	
	# Set person indexes.
	df['person_index'] = -999
	for p_id, p_idx in person_dic.items():
		cond = df['person_id']==p_id
		df.loc[cond, 'person_index'] = p_idx

	# Save annotation data as a csv file.
	cols = ['person_index', 'person_id', 'camera_id', 'sequence_id', 'frame_no', 'dpm_bbox_no', 'mode', 'image_name', 'image_path']
	df = df[cols]
	df.to_csv(anno_path, index=False)


def parse_args():
	pass
	"""
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument('--dataset_name', default='Market1501')
	arg_parser.add_argument('--data_dir', default='D:/workspace/datasets/Market-1501-v15.09.15/')
	arg_parser.add_argument('--anno_dir', default='../../data/annos/')

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.anno_dir, exist_ok=True)
	#os.makedirs(args.model_ckpt_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.anno_dir)

	return args
	"""


if __name__ == "__main__":
	make_csv()