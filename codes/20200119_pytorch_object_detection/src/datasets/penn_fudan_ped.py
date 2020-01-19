import os
import numpy as np
import torch
from PIL import Image
import cv2

import sys
#sys.path.append('../vision/references/detection/')
sys.path.append('./vision/references/detection/')
import transforms as T
import utils

class PennFudanPedDataset(object):
	def __init__(self, root, transforms):
		self.root = root
		self.transforms = transforms
		# load all image files, sorting them to
		# ensure that they are aligned
		self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
		self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

	def __getitem__(self, idx):
		# load images ad masks
		img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
		mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
		img = Image.open(img_path).convert("RGB")
		# note that we haven't converted the mask to RGB,
		# because each color corresponds to a different instance
		# with 0 being background
		mask = Image.open(mask_path)
		# convert the PIL Image into a numpy array
		mask = np.array(mask)
		# instances are encoded as different colors
		obj_ids = np.unique(mask)
		# first id is the background, so remove it
		obj_ids = obj_ids[1:]

		# split the color-encoded mask into a set
		# of binary masks
		masks = mask == obj_ids[:, None, None]

		# get bounding box coordinates for each mask
		num_objs = len(obj_ids)
		boxes = []
		for i in range(num_objs):
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])

		# convert everything into a torch.Tensor
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		# there is only one class
		labels = torch.ones((num_objs,), dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)

		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		# suppose all instances are not crowd
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.imgs)


def get_dataset(data_dir):
	# use our dataset and defined transformations
	dataset_train = PennFudanPedDataset(data_dir, get_transform(train=True))
	dataset_test = PennFudanPedDataset(data_dir, get_transform(train=False))

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset_train)).tolist()
	dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	train_data_loader = torch.utils.data.DataLoader(
		dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

	test_data_loader = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

	return train_data_loader, test_data_loader


def get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


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



if __name__ == "__main__":
	test()


