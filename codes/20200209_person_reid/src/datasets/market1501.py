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


def make_anno_data(data_root_dir, anno_path):
	# Set data directories.
	train_dir = data_root_dir + 'bounding_box_train/'
	gallery_dir = data_root_dir + 'bounding_box_test/'
	query_dir = data_root_dir + 'query/'

	# Get image names.
	train_img_names = sorted([d for d in os.listdir(train_dir) if d.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')])
	gallery_img_names = sorted([d for d in os.listdir(gallery_dir) if d.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')])
	query_img_names = sorted([d for d in os.listdir(query_dir) if d.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')])
	
	# Organize anntation data.
	train_list = __org_data(train_dir, train_img_names, 'train')
	gallery_list = __org_data(gallery_dir, gallery_img_names, 'gallery')
	query_list = __org_data(query_dir, query_img_names, 'query')
	data_list = train_list + gallery_list + query_list
	df = pd.DataFrame(data_list)
	df = __add_person_index(df)

	# Save DataFrame.
	__debug(df)
	df.to_csv(anno_path, index=False)
	print('Saved annotation data of Market1501 to {}'.format(anno_path))
	

def __org_data(data_dir, img_names, mode):
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


def __add_person_index(df):
	# Make person ID and index dictonary.
	train_person_ids = sorted(set(df.loc[df['mode']=='train', 'person_id'].values))
	train_person_dic = {train_person_ids[i]: i for i in range(len(train_person_ids))}
	
	# Set person indexes.
	df['person_index'] = -999
	for p_id, p_idx in train_person_dic.items():
		cond = df['person_id']==p_id
		df.loc[cond, 'person_index'] = p_idx

	# Save annotation data as a csv file.
	cols = ['person_index', 'person_id', 'camera_id', 'sequence_id', 'frame_no', 'dpm_bbox_no', 'mode', 'image_name', 'image_path']
	df = df[cols]
	
	return df


def __debug(df):
	# Check mode type.
	assert set(df['mode'].values) == set(['train', 'gallery', 'query'])

	# Check the number of person indexes.
	person_indexes = sorted(set(df.loc[df['mode']=='train', 'person_index'].values))
	assert len(person_indexes) == person_indexes[-1]+1


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
	# cf. https://github.com/GNAYUOHZ/ReID-MGN/blob/master/data.py
	transform_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	train_set = Market1501(anno_path, 'train', transforms=transform_train)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=n_batch, shuffle=True, num_workers=0)
	gallery_set = Market1501(anno_path, 'gallery', transforms=transform_test)
	gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=n_batch, shuffle=False, num_workers=0)
	query_set = Market1501(anno_path, 'query', transforms=transform_test)
	query_loader = torch.utils.data.DataLoader(query_set, batch_size=1, shuffle=False, num_workers=0)
	df = pd.read_csv(anno_path)
	class_names = sorted(list(set(df.loc[df['mode']=='train', 'person_index'].values)))
		
	return train_loader, gallery_loader, query_loader, class_names

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description='')

	arg_parser.add_argument('--data_root_dir', default='D:/workspace/datasets/Market-1501-v15.09.15/')
	arg_parser.add_argument('--anno_dir', default='../../data/annos/')
	arg_parser.add_argument('--anno_path', default='../../data/annos/anno_market1501.csv')
	
	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.anno_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_root_dir)
	assert os.path.exists(args.anno_dir)

	return args


if __name__ == "__main__":
	args = parse_args()
	make_anno_data(args.data_root_dir, args.anno_path)
