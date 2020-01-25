import os
import argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
	args = parse_args()
	df = pd.read_csv(args.anno_path)
	label_names = sorted(set(df['label_name'].values.tolist()))
	colormap = get_colormap(label_names, args.colormap_name)
	#print(colormap)

	img_paths = sorted(set(df['image_path'].values.tolist()))
	for img_path in img_paths:
		assert os.path.exists(img_path)
		cond = df['image_path']==img_path
		df_img = df.loc[cond, ['label_name', 'xmin', 'ymin', 'xmax', 'ymax']]
		annos = df_img.to_dict('record')
		drawn_anno_img_path = None
		if args.drawn_anno_dir != None:
			drawn_anno_img_path = args.drawn_anno_dir + img_path.split('/')[-1]
		visual_anno(img_path, annos, colormap, drawn_anno_img_path)


def get_colormap(label_names, colormap_name):
	colormap = {}	
	cmap = plt.get_cmap(colormap_name)
	for i in range(len(label_names)):
		rgb = [int(d) for d in np.array(cmap(float(i)/len(label_names)))*255][:3]
		colormap[label_names[i]] = tuple(rgb)

	return colormap


def visual_anno(img_path, annos, colormap, drawn_anno_img_path=None):
	# Draw annotion data on image.
	img = cv2.imread(img_path)
	for a in annos:
		color = colormap[a['label_name']]
		cv2.rectangle(img, (a['xmin'], a['ymin']), (a['xmax'], a['ymax']), color, 2)
		text = '{}'.format(a['label_name'])
		cv2.putText(img, text, (a['xmin'], a['ymin']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

	# Save or show an image.
	if drawn_anno_img_path != None:
		cv2.imwrite(drawn_anno_img_path, img)
	else:
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument('--anno_path', default='../data/annos/anno_voc.csv')
	arg_parser.add_argument("--colormap_name", default='gist_rainbow')
	arg_parser.add_argument('--drawn_anno_dir', default='../experiments/results/drawn_anno_images/')

	args = arg_parser.parse_args()

	# Make directories.
	os.makedirs(args.drawn_anno_dir, exist_ok=True)

	# Validate paths.
	assert os.path.exists(args.anno_path)
	assert os.path.exists(args.drawn_anno_dir)

	return args


if __name__ == "__main__":
	main()
