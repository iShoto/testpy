import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import cv2

from datasets import penn_fudan_ped
import models

import sys
sys.path.append('./vision/references/detection/')
import engine, utils


def main():
	args = parse_args()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	num_classes = 2
	train_data_loader, test_data_loader = penn_fudan_ped.get_dataset(args.anno_path)

	print('Loading a model from {}'.format(args.model_weight_path))
	model = models.get_fasterrcnn_resnet50(num_classes)
	model.load_state_dict(torch.load(args.model_weight_path))
	model = model.to(device)

	detect_objects(model, test_data_loader, device)
	#calc_score(model, test_data_loader, device)
	#draw_detection_results(model, test_data_loader, device, args)


def detect_objects(model, test_data_loader, device):
	model.eval()

	# Draw detection results to test images.
	for batch_idx, (inputs, targets) in enumerate(test_data_loader):
		# Display progress.
		precentage = int((batch_idx+1)/len(test_data_loader)*100)
		print('\rDrawing detection results to images... {:>3}%'.format(precentage), end='')

		# Prepare output image data.
		imgs = [img for img in inputs]
		img_ids = [target['image_id'].numpy().tolist()[0] for target in targets]

		# Detect objects
		inputs = [img.to(device) for img in inputs]
		outputs = model(inputs)
		for i in range(len(outputs)):
			# Organize output image.
			img = imgs[i]
			img = img.mul(255).permute(1,2,0).byte().numpy()
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			# Set output image index.
			img_id = img_ids[i]

			# Organize detection results.
			scores = [round(d,3) for d in outputs[i]['scores'].tolist()]
			boxes = [[int(round(b)) for b in box] for box in outputs[i]['boxes'].tolist()]
			labels = outputs[i]['labels'].tolist()
			assert len(scores) == len(boxes) == len(labels)

			# Draw detection results to the image.
			for j in range(len(scores)):
				if scores[j] < args.score_thresh:
					continue
				cv2.rectangle(img, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), (0,255,0), 2)
				text = '{}: {}%'.format(labels[j], int(scores[j]*100))
				cv2.putText(img, text, (boxes[j][0], boxes[j][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

			# Resize and save the image.
			if args.visual_img_half == True:
				h,w,c = img.shape
				img = cv2.resize(img, (int(w*0.5), int(h*0.5)))
			cv2.imwrite(args.visual_img_dir+'visual_img_{}.png'.format(str(img_id).zfill(6)), img)

	print('')
	print('Done.')



def calc_score(model, test_data_loader, device):
	engine.evaluate(model, test_data_loader, device=device)


def draw_results(model, test_data_loader, device, args):
	model.eval()

	# Draw detection results to test images.
	for batch_idx, (inputs, targets) in enumerate(test_data_loader):
		# Display progress.
		precentage = int((batch_idx+1)/len(test_data_loader)*100)
		print('\rDrawing detection results to images... {:>3}%'.format(precentage), end='')

		# Prepare output image data.
		imgs = [img for img in inputs]
		img_ids = [target['image_id'].numpy().tolist()[0] for target in targets]

		# Detect objects
		inputs = [img.to(device) for img in inputs]
		outputs = model(inputs)
		for i in range(len(outputs)):
			# Organize output image.
			img = imgs[i]
			img = img.mul(255).permute(1,2,0).byte().numpy()
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			# Set output image index.
			img_id = img_ids[i]

			# Organize detection results.
			scores = [round(d,3) for d in outputs[i]['scores'].tolist()]
			boxes = [[int(round(b)) for b in box] for box in outputs[i]['boxes'].tolist()]
			labels = outputs[i]['labels'].tolist()
			assert len(scores) == len(boxes) == len(labels)

			# Draw detection results to the image.
			for j in range(len(scores)):
				if scores[j] < args.score_thresh:
					continue
				cv2.rectangle(img, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), (0,255,0), 2)
				text = '{}: {}%'.format(labels[j], int(scores[j]*100))
				cv2.putText(img, text, (boxes[j][0], boxes[j][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

			# Resize and save the image.
			if args.visual_img_half == True:
				h,w,c = img.shape
				img = cv2.resize(img, (int(w*0.5), int(h*0.5)))
			cv2.imwrite(args.visual_img_dir+'visual_img_{}.png'.format(str(img_id).zfill(6)), img)

	print('')
	print('Done.')


def draw_results_core(img_path, annos, colormap, drawn_img_path=None, half_img=False):
	# Draw annotion data on image.
	img = cv2.imread(img_path)
	for a in annos:
		color = colormap[a['label_name']]
		cv2.rectangle(img, (a['xmin'], a['ymin']), (a['xmax'], a['ymax']), color, 2)
		text = '{}'.format(a['label_name'])
		cv2.putText(img, text, (a['xmin'], a['ymin']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

	# Save or show an image.
	if drawn_img_path != None:
		# Resize the image.
		if args.half_img == True:
			h,w,c = img.shape
			img = cv2.resize(img, (int(w*0.5), int(h*0.5)))
		# Save the image.
		cv2.imwrite(drawn_img_path, img)
	else:
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	# Dataset
	arg_parser.add_argument("--dataset_name", default='PennFudanPed')
	arg_parser.add_argument("--data_dir", default='D:/workspace/datasets/PennFudanPed/')
	arg_parser.add_argument('--anno_path', default='../data/annos/anno_penn-fudan-ped.csv')

	# Model
	arg_parser.add_argument("--model_name", default='FasterRCNN-ResNet50')
	arg_parser.add_argument("--model_weight_path", default='../experiments/models/checkpoints/PennFudanPed_FasterRCNN-ResNet50_epoch=1.pth')
	
	# Results
	arg_parser.add_argument('--visual_img_dir', default='../experiments/results/visual_images/')
	arg_parser.add_argument('--visual_img_half', default=1, type=int, help='Resize images half. 0 is False, 1 is True.')

	# Others 
	arg_parser.add_argument('--score_thresh', default=0.75, type=float)
	
	args = arg_parser.parse_args()

	# Make directories.
	os.makedirs(args.visual_img_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.data_dir)
	assert os.path.exists(args.anno_path)
	assert os.path.exists(args.model_weight_path)
	assert os.path.exists(args.visual_img_dir)

	return args


if __name__ == "__main__":
	main()
