import os
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
import random

import sys
sys.path.append('./vision/references/detection/')
#sys.path.append('../vision/references/detection/')
import transforms as T
import utils

class PennFudanPedDataset(object):
	def __init__(self, anno_path, data_type, transforms=None):
		df_all = pd.read_csv(anno_path)
		self.df = df_all[df_all['data_type']==data_type].reset_index(drop=True)
		img_paths = sorted(set(self.df['image_path'].values.tolist()))
		self.img_dict = {i: img_paths[i] for i in range(len(img_paths))}
		self.transforms = transforms

	def __getitem__(self, idx):
		# Filter data
		img_path = self.img_dict[idx]
		df = self.df.loc[self.df['image_path']==img_path, :]

		# Image
		img_path = df['image_path'].values[0]
		assert os.path.exists(img_path)
		img = Image.open(img_path).convert('RGB')

		# Target
		target = {}
		boxes = []
		labels = []
		for r in df.to_dict('record'):
			boxes.append([r['xmin'], r['ymin'], r['xmax'], r['ymax']])
			labels.append(r['label'])
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.int64)
		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros(len(labels,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		#target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(set(self.df['image_path'].values.tolist()))


def get_dataset(anno_path):
	# use our dataset and defined transformations
	dataset_train = PennFudanPedDataset(anno_path, 'train', __get_transform(train=True))
	dataset_test = PennFudanPedDataset(anno_path, 'test', __get_transform(train=False))

	# define training and validation data loaders
	train_data_loader = torch.utils.data.DataLoader(
		dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

	test_data_loader = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

	return train_data_loader, test_data_loader


def __get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


def get_image_id_dict(anno_path, data_mode='test'):
	dataset_test = PennFudanPedDataset(anno_path, data_mode, __get_transform(train=False))
	img_id_dict = dataset_test.img_dict
	
	return img_id_dict


def test():
	# Change the path!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# to sys.path.append('../vision/references/detection/')
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	data_dir = 'D:/workspace/datasets/PennFudanPed/'
	train_data_loader, test_data_loader = get_dataset(data_dir)
	for batch_idx, (inputs, targets) in enumerate(train_data_loader):
		inputs = list(img.numpy() for img in inputs)
		targets = [{k: v.numpy() for k, v in t.items() if k in ('boxes', 'labels')} for t in targets]

		img = inputs[0]*255
		img = np.array(img, dtype=np.uint8)
		img = img.transpose(1,2,0)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		bboxes = targets[0]['boxes']
		labels = targets[0]['labels']
		assert len(bboxes)==len(labels)

		for i in range(len(bboxes)):
			bbox = bboxes[i]
			label = labels[i]
			xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
			cv2.putText(img, str(label), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
			
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def make_csv(data_dir, anno_path):
	# Make an annotation file with csv file.
	
	# Get paths.
	img_paths = sorted([data_dir+'PNGImages/'+d for d in os.listdir(data_dir+'PNGImages/')])
	mask_paths = sorted([data_dir+'PedMasks/'+d for d in os.listdir(data_dir+'PedMasks/')])

	# Debug.
	for img_path in img_paths:
		assert os.path.exists(img_path)
	for mask_path in mask_paths:
		assert os.path.exists(mask_path)

	# Debug
	for i in range(len(img_paths)):
		img_id = img_paths[i].split('/')[-1].split('.')[0]
		mask_id = mask_paths[i].split('/')[-1]
		assert img_id in mask_id, 'Image ID {} is not contained in mask ID {}'.format(img_id, mask_id)

	# Set annotation data.
	lis = []
	for i in range(len(img_paths)):
		img_path = img_paths[i]
		mask_path = mask_paths[i]
		mask = Image.open(mask_path)
		mask = np.array(mask)
		obj_ids = np.unique(mask)
		obj_ids = obj_ids[1:]
		masks = mask == obj_ids[:, None, None]

		num_objs = len(obj_ids)
		for j in range(num_objs):
			dic = {}
			pos = np.where(masks[j])
			dic['xmin'] = np.min(pos[1])
			dic['xmax'] = np.max(pos[1])
			dic['ymin'] = np.min(pos[0])
			dic['ymax'] = np.max(pos[0])
			dic['image_path'] = img_path
			dic['label'] = 1
			lis.append(dic)

	# Set training or test.
	df = pd.DataFrame(lis)
	img_paths = set(df['image_path'].values.tolist())
	test_img_paths = set(random.sample(img_paths, 50))
	train_img_paths = img_paths - test_img_paths
	df['data_type'] = 'train'
	cond = df['image_path'].isin(list(test_img_paths))
	df.loc[cond, 'data_type'] = 'test'

	# Save anntation file.
	cols = ['label', 'xmax', 'xmin', 'ymax', 'ymin', 'data_type', 'image_path']
	df = df[cols]
	df.to_csv(anno_path, index=False)
	print('Saved an annotation file with csv format at {}'.format(anno_path))


if __name__ == "__main__":
	test()
	

