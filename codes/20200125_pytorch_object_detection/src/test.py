import torch
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
"""

import os
import argparse
import cv2
import pandas as pd
from tqdm import trange
import shutil
import matplotlib.pyplot as plt
import numpy as np

from datasets import penn_fudan_ped
import models

import sys
#sys.path.append('./vision/references/detection/')
#import engine, utils
sys.path.append('./mAP/')
import main_ex


def main():
	args = parse_args()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	num_classes = 2
	train_data_loader, test_data_loader = penn_fudan_ped.get_dataset(args.data_anno_path)

	print('Loading a model from {}'.format(args.model_weight_path))
	model = models.get_fasterrcnn_resnet50(num_classes, pretrained=False)
	model.load_state_dict(torch.load(args.model_weight_path))
	model = model.to(device)

	#detect_objects(args.data_anno_path, test_data_loader, model, device, args.det_result_path)
	#calc_score(args.data_anno_path, args.anno_text_dir, args.det_result_path, args.det_text_dir)
	draw_gt_n_det(args.data_anno_path, args.det_result_path, args.drawn_img_dir)


def detect_objects(data_anno_path, test_data_loader, model, device, det_result_path):
	model.eval()

	# Get image and its ID dictionary.
	dataset_test = penn_fudan_ped.PennFudanPedDataset(data_anno_path, 'test')
	img_id_dict = dataset_test.img_dict
	
	# Draw detection results to test images.
	lis = []
	for batch_idx, (inputs, targets) in enumerate(test_data_loader):
		# Display progress.
		precentage = int((batch_idx+1)/len(test_data_loader)*100)
		print('\rDetecting objects... {:>3}%'.format(precentage), end='')

		# Prepare output image data.
		img_ids = [target['image_id'].numpy().tolist()[0] for target in targets]

		# Detect objects.
		inputs = [img.to(device) for img in inputs]
		outputs = model(inputs)
		for i in range(len(outputs)):
			# Organize detection results.
			img_path = img_id_dict[img_ids[i]]
			scores = [round(d,3) for d in outputs[i]['scores'].tolist()]
			boxes = [[int(round(b)) for b in box] for box in outputs[i]['boxes'].tolist()]
			labels = outputs[i]['labels'].tolist()
			assert len(scores) == len(boxes) == len(labels)

			# Set detection results.
			for j in range(len(scores)):
				dic = {}
				dic['image_path'] = img_path
				dic['label'] = labels[j]
				dic['score'] = scores[i]
				dic['xmin'] = boxes[j][0]
				dic['ymin'] = boxes[j][1]
				dic['xmax'] = boxes[j][2]
				dic['ymax'] = boxes[j][3]
				lis.append(dic)
	print('')

	# Make DataFrame of detection results.
	df = pd.DataFrame(lis)
	df = df[['label', 'score', 'xmin', 'ymin', 'xmax', 'ymax', 'image_path']]
	print('========================== DETECTION RESULTS ==========================')
	print(df.head(10).to_string())
	
	# Save the DataFrame with csv format.
	df.to_csv(det_result_path, index=False)
	print('Detection results saved to {}'.format(det_result_path))


def calc_score(gt_csv_path, gt_text_dir, det_csv_path, det_text_dir):
	# Use https://github.com/Cartucho/mAP

	make_text_files(gt_csv_path, gt_text_dir, file_type='gt')
	make_text_files(det_csv_path, det_text_dir, file_type='det')
	main_ex.main()


def make_text_files(csv_path, text_dir, file_type):
	# Use library https://github.com/Cartucho/mAP
	# Get image paths.
	df = pd.read_csv(csv_path)
	if file_type == 'gt':
		cond_test = df['data_type']=='test'
		img_paths = sorted(set(df.loc[cond_test, 'image_path'].values.tolist()))
	elif file_type == 'det':
		img_paths = sorted(set(df['image_path'].values.tolist()))
	
	# Make text files.
	for i in trange(len(img_paths), desc='Making {} text files'.format(file_type)):
		# Set text.
		text = ''
		img_path = img_paths[i]
		cond_img = df['image_path']==img_path
		records = df.loc[cond_img, :].to_dict('record')
		for r in records:
			if file_type == 'gt':
				text += '{} {} {} {} {}\n'.format(r['label'], r['xmin'], r['ymin'], r['xmax'], r['ymax'])
			elif file_type == 'det':
				text += '{} {} {} {} {} {}\n'.format(r['label'], round(r['score'], 3), r['xmin'], r['ymin'], r['xmax'], r['ymax'])
		text = text.strip()

		# Save text file.
		text_name = img_path.split('/')[-1].split('.')[0]+'.txt'
		f = open(text_dir+text_name, 'w')
		f.write(text)
		f.close()



def draw_gt_n_det(gt_csv_path, det_csv_path, drawn_img_dir):
	# Use library https://github.com/Cartucho/mAP
	# Get image paths.
	df_gt = pd.read_csv(gt_csv_path)
	df_det = pd.read_csv(det_csv_path)
	df_gt['data_type'] = 'gt'
	df_det['data_type'] = 'det'

	data_types = ['gt', 'det']
	colormap = get_colormap(data_types)

	img_paths = sorted(set(df_det['image_path'].values.tolist()))
	for i in trange(len(img_paths), desc=''):
		img_path = img_paths[i]
		cond_gt = df_gt['image_path']==img_path
		cond_det = df_det['image_path']==img_path
		gt_annos = df_gt.loc[cond_gt, :].to_dict('record')
		det_annos = df_det.loc[cond_det, :].to_dict('record')
		annos = gt_annos + det_annos
		drawn_img_path = drawn_img_dir + img_path.split('/')[-1]
		draw_gt_n_det_core(img_path, annos, colormap, drawn_img_path=drawn_img_path, half_img=False)
		

def get_colormap(label_names, colormap_name='gist_rainbow'):
	colormap = {}   
	cmap = plt.get_cmap(colormap_name)
	for i in range(len(label_names)):
		rgb = [int(d) for d in np.array(cmap(float(i)/len(label_names)))*255][:3]
		colormap[label_names[i]] = tuple(rgb)

	return colormap


def draw_gt_n_det_core(img_path, annos, colormap, drawn_img_path=None, half_img=False):
	# Draw annotion data on image.
	img = cv2.imread(img_path)
	for a in annos:
		color = colormap[a['data_type']]
		cv2.rectangle(img, (a['xmin'], a['ymin']), (a['xmax'], a['ymax']), color, 2)
		text = '{}'.format(a['data_type'])
		if a['data_type'] == 'det':
			text = '{}: {}%'.format(a['data_type'], int(a['score']*100))
		cv2.putText(img, text, (a['xmin'], a['ymin']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

	# Save or show an image.
	if drawn_img_path != None:
		# Resize the image.
		if half_img == True:
			h,w,c = img.shape
			img = cv2.resize(img, (int(w*0.5), int(h*0.5)))
		# Save the image.
		cv2.imwrite(drawn_img_path, img)
	else:
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def remake_dir(directory):
	if os.path.exists(directory):
		shutil.rmtree(directory)
	os.makedirs(directory)


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	# Dataset
	arg_parser.add_argument("--dataset_name", default='PennFudanPed')
	arg_parser.add_argument("--data_dir", default='D:/workspace/datasets/PennFudanPed/')
	arg_parser.add_argument('--data_anno_path', default='../data/annos/anno_penn-fudan-ped.csv')
	arg_parser.add_argument('--anno_text_dir', default='./mAP/input/ground-truth/')

	# Model
	arg_parser.add_argument("--model_name", default='FasterRCNN-ResNet50')
	arg_parser.add_argument("--model_weight_path", default='../experiments/models/PennFudanPed_FasterRCNN-ResNet50_epoch=10.pth')
	
	# Results
	arg_parser.add_argument('--det_result_dir', default='../experiments/results/detections/')
	arg_parser.add_argument('--det_result_path', default='../experiments/results/detections/dets.csv')
	arg_parser.add_argument('--det_text_dir', default='./mAP/input/detection-results/')
	arg_parser.add_argument('--drawn_img_dir', default='../experiments/results/drawn_images/')
	arg_parser.add_argument('--visual_img_half', default=1, type=int, help='Resize images half. 0 is False, 1 is True.')

	# Others 
	arg_parser.add_argument('--score_thresh', default=0.75, type=float)
	
	args = arg_parser.parse_args()

	# Make directories.
	os.makedirs(args.det_result_dir, exist_ok=True)
	os.makedirs(args.drawn_img_dir, exist_ok=True)
	remake_dir(args.anno_text_dir)
	remake_dir(args.det_text_dir)
	
	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.data_anno_path)
	assert os.path.exists(args.model_weight_path)
	assert os.path.exists(args.det_result_dir)
	assert os.path.exists(args.drawn_img_dir)

	return args


if __name__ == "__main__":
	main()
