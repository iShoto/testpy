# coding: utf-8

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
from progressbar import ProgressBar 
import shutil

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label


class Image_Clustering:
	def __init__(self, n_clusters=50, video_file='IMG_2140.MOV', image_file_temp='img_%s.png', input_video=False):
		self.n_clusters = n_clusters            # The number of cluster
		self.video_file = video_file            # Input video file name
		self.image_file_temp = image_file_temp  # Image file name template
		self.input_video = input_video          # If input data is a video


	def main(self):
		if self.input_video == True:
			self.video_2_frames()
		self.label_images()
		self.classify_images()
		
		#self.vgg16_recoginition()
		#self.vgg16_feature_extraction()


	def video_2_frames(self):
		print('Video to frames...')
		cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)

		# Remove and make a directory.
		if os.path.exists(TARGET_IMAGES_DIR):
			shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
		if not os.path.exists(TARGET_IMAGES_DIR):
			os.makedirs(TARGET_IMAGES_DIR)	# Make a directory

		i = 0
		while(cap.isOpened()):
			flag, frame = cap.read()  # Capture frame-by-frame
			if flag == False:
				break  # A frame is not left
			cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame
			i += 1
			print('Save', TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6))
			#cv2.imshow('frame', frame)
			#if cv2.waitKey(1) & 0xFF == ord('q'):
			#	break

		cap.release()  # When everything done, release the capture
		#cv2.destroyAllWindows()
		print('')


	def label_images(self):
		print('Label images...')

		# Load a model
		model = VGG16(weights='imagenet', include_top=False)
	
		# Get images
		images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.png', '.jpg']]
		assert(len(images)>0)
		
		X = []
		pb = ProgressBar(max_value=len(images))
		for i in range(len(images)):
			# Extract image features
			feat = self.__feature_extraction(model, TARGET_IMAGES_DIR+images[i])
			X.append(feat)
			pb.update(i)  # Update progressbar

		# Clutering images by k-means++
		X = np.array(X)
		kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
		print('')
		print('labels:')
		print(kmeans.labels_)
		print('')
		
		# Merge images and labels
		df = pd.DataFrame({'image': images, 'label': kmeans.labels_})
		df.to_csv(DATA_DIR+IMAGE_LABEL_FILE, index=False)


	def __feature_extraction(self, model, img_path):
		img = image.load_img(img_path, target_size=(224, 224))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels

		feat = model.predict(x)  # Get image features
		print(feat)
		print(feat.shape)
		feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
		print(feat)
		print(feat.shape)

		1/0

		return feat


	def classify_images(self):
		print('Classify images...')

		# Get labels and images
		df = pd.read_csv(DATA_DIR+IMAGE_LABEL_FILE)
		labels = list(set(df['label'].values))
		
		# Delete images which were clustered before
		if os.path.exists(CLUSTERED_IMAGES_DIR):
			shutil.rmtree(CLUSTERED_IMAGES_DIR)

		for label in labels:
			print('Copy and paste label %s images.' % label)

			# Make directories named each label
			new_dir = CLUSTERED_IMAGES_DIR + str(label) + '/'
			if not os.path.exists(new_dir):
				os.makedirs(new_dir)

			# Copy images to the directories
			clustered_images = df[df['label']==label]['image'].values
			for ci in clustered_images:
				src = TARGET_IMAGES_DIR + ci
				dst = CLUSTERED_IMAGES_DIR + str(label) + '/' + ci
				shutil.copyfile(src, dst)

		print('')






	def vgg16_recoginition(self, img_path):
		model = VGG16(weights='imagenet')
		# model.summary()

		img = image.load_img(img_path, target_size=(224, 224))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples.
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels.
		
		preds = model.predict(x)
		results = decode_predictions(preds, top=5)[0]
		for result in results:
			print(result)

		
if __name__ == "__main__":
	Image_Clustering().main()
