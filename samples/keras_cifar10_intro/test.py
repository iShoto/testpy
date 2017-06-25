# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
import pandas as pd
import time
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split

from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

N_CLASS = 10
N_EPOCH = 50  # 100
BATCH_SIZE = 128
INPUT_DIM = (32, 32, 3)
DATA_AUGMENTATION = True
IDG_PARAM = {'featurewise_center': False,
			 'samplewise_center': False,
			 'featurewise_std_normalization': False,
			 'samplewise_std_normalization': False,
			 'zca_whitening': True,  # False
			 'rotation_range': 0.,
			 'width_shift_range': 0.1, # 0.,
			 'height_shift_range': 0.1, # 0.,
			 'shear_range': 0.,
			 'zoom_range': 0.,
			 'channel_shift_range': 0.,
			 'fill_mode': 'nearest',
			 'cval': 0.,
			 'horizontal_flip': True,
			 'vertical_flip': False,
			 'rescale': None,
			 'preprocessing_function': None
}

DIR = './result/'
MODEL_FILE = 'model.json'
WEIGHT_FILE = 'weights.h5'
HISTORY_DATA_FILE = 'history.csv'
HISTORY_IMAGE_FILE = 'history.jpg'
PARAM_EVAL_FILE = 'param_eval.csv'


class Test:
	def __init__(self):
		"""
		data augmentation
		normalize 
		zca whitening
		make validation data from training data
		change learning rate on a way
		"""
		pass


	def main(self):
		# Training
		start = time.clock()
		data = self.get_data()
		model = self.design_model(data[0])
		result = self.train_model(data, model)
		self.save(result)
		print('Training Time: %s min' % round((time.clock()-start)/60., 1))
		print('')

		# Test
		self.test_model(data)


	def get_data(self):
		# Load CIFAR-10
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
		self.__draw_sample_images(X_train, y_train)

		# Normalize data
		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train /= 255.0
		X_test /= 255.0

		# Onehot label
		Y_train = np_utils.to_categorical(y_train, N_CLASS)
		Y_test = np_utils.to_categorical(y_test, N_CLASS)

		print('X_train.shape:', X_train.shape, 'Y_train.shape:', Y_train.shape)
		print('X_test.shape:', X_test.shape, 'Y_test.shape:', Y_test.shape)

		return X_train, Y_train, X_test, Y_test


	def __draw_sample_images(self, X_train, y_train, stdout=False):
		# Set background color to white
		fig = plt.figure()
		fig.patch.set_facecolor('white')

		# Draw sample images
		n_class = 10
		pos = 1
		for target_class in range(n_class):
			# Get index list of a class
			target_idx = []
			for i in range(len(y_train)):
				if y_train[i][0] == target_class:
					target_idx.append(i)

			# Draw random ten images for each class
			np.random.shuffle(target_idx)
			for idx in target_idx[:10]:
				img = toimage(X_train[idx])
				plt.subplot(10, 10, pos)
				plt.imshow(img)
				plt.axis('off')
				pos += 1

		plt.savefig(DIR+'cifar10.jpg', dpi=100)
		if stdout == True:
			plt.show()


	def design_model(self, X_train):
		# Initialize
		model = Sequential()
		
		# (Conv -> Relu) * 2 -> Pool -> Dropout
		model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 3, 3, border_mode='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (Conv -> Relu) * 2 -> Pool -> Dropout
		model.add(Convolution2D(64, 3, 3, border_mode='same'))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, border_mode='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Flatten
		model.add(Flatten())  # 6*6*64
		
		# FC -> Relu -> Dropout
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		# FC -> Softmax
		model.add(Dense(N_CLASS))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])

		# output model summary!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# File ".\test.py", line 13, in <module>
		# 	  from keras.utils.visualize_util import plot
		# ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
		#model.summary()
		#plot(model, show_shapes=True, to_file=os.path.join(DIR, 'model.png'))
		model.summary()

		return model


	def train_model(self, data, model):
		X_train, Y_train, X_test, Y_test = data

		if not DATA_AUGMENTATION:
			print('Not using data augmentation')

			# Train the model
			history = model.fit(X_train, Y_train,
								batch_size=BATCH_SIZE,
								nb_epoch=N_EPOCH,
								verbose=1,
								validation_data=(X_test, Y_test),
								shuffle=True)
		else:
			print('Using real-time data augmentation')

			# Make a generator for training data
			train_datagen = ImageDataGenerator(featurewise_center=IDG_PARAM['featurewise_center'],
											   samplewise_center=IDG_PARAM['samplewise_center'],
											   featurewise_std_normalization=IDG_PARAM['featurewise_std_normalization'],
											   samplewise_std_normalization=IDG_PARAM['samplewise_std_normalization'],
											   zca_whitening=IDG_PARAM['zca_whitening'],
											   rotation_range=IDG_PARAM['rotation_range'],
											   width_shift_range=IDG_PARAM['width_shift_range'],
											   height_shift_range=IDG_PARAM['height_shift_range'],
											   shear_range=IDG_PARAM['shear_range'],
											   zoom_range=IDG_PARAM['zoom_range'],
											   channel_shift_range=IDG_PARAM['channel_shift_range'],
											   fill_mode=IDG_PARAM['fill_mode'],
											   cval=IDG_PARAM['cval'],
											   horizontal_flip=IDG_PARAM['horizontal_flip'],
											   vertical_flip=IDG_PARAM['vertical_flip'],
											   rescale=IDG_PARAM['rescale'],
											   preprocessing_function=IDG_PARAM['preprocessing_function'])
			train_datagen.fit(X_train)
			train_generator = train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)

			# Make a generator for test data
			test_datagen = ImageDataGenerator(zca_whitening=IDG_PARAM['zca_whitening'])
			test_datagen.fit(X_test)
			test_generator = test_datagen.flow(X_test, Y_test)

			# Train the model
			history = model.fit_generator(train_generator,
										  samples_per_epoch=X_train.shape[0],
										  nb_epoch=N_EPOCH,
										  validation_data=test_generator,
										  nb_val_samples=X_test.shape[0])

			# Evaluate the model
			if not DATA_AUGMENTATION:
				loss, acc = model.evaluate(X_test, Y_test, verbose=0)
			else:
				loss, acc = model.evaluate_generator(test_generator, val_samples=X_test.shape[0])

			print('Test loss: %s, Test acc: %s' % (loss, acc))

		result = {'model': model, 'history': history, 'loss': loss, 'acc': acc}

		return result


	def save(self, result):
		"""
		Save model, weight, history, parameter and evaluation
		"""
		model = result['model']
		history = result['history']
		loss = result['loss']
		acc = result['acc']

		# Model
		model_json = model.to_json()
		
		# Weight
		with open(os.path.join(DIR, MODEL_FILE), 'w') as json_file:
			json_file.write(model_json)
		model.save_weights(os.path.join(DIR, WEIGHT_FILE))
		
		# History
		self.__save_history(history)
		self.__plot_history(history)
		
		# Param and evaluation
		dic = IDG_PARAM
		dic.update({'n_epoch': N_EPOCH, 'batch_size': BATCH_SIZE, 'loss': loss, 'acc': acc})
		if os.path.exists(DIR+PARAM_EVAL_FILE):
			df = pd.read_csv(DIR+PARAM_EVAL_FILE)
			df = pd.concat([df, pd.DataFrame([dic])])
		else:
			df = pd.DataFrame([dic])
		df.to_csv(DIR+PARAM_EVAL_FILE, index=False)


	def __save_history(self, history, stdout=False):
		df = pd.DataFrame()
		df['train_loss'] = history.history['loss']
		df['train_acc'] = history.history['acc']
		df['valid_loss'] = history.history['val_loss']
		df['valid_acc'] = history.history['val_acc']
		df.to_csv(DIR+HISTORY_DATA_FILE, index=False)
		if stdout == True:
			print(df)
		

	def __plot_history(self, history, stdout=False):
		# Set background color to white
		fig = plt.figure()
		fig.patch.set_facecolor('white')
		fig.set_size_inches(16.0, 9.0, forward=True)

		# Plot accuracy history
		plt.subplot(1, 2, 1)
		plt.plot(history.history['acc'], "o-", label="train_acc")
		plt.plot(history.history['val_acc'], "o-", label="valid_acc")
		plt.title('model accuracy')
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.xlim(0)
		plt.ylim(0, 1)
		plt.legend(loc="lower right")

		# Plot loss history
		plt.subplot(1, 2, 2)
		plt.plot(history.history['loss'], "o-", label="train_loss",)
		plt.plot(history.history['val_loss'], "o-", label="valid_loss")
		plt.title('model loss')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.xlim(0)
		plt.ylim(0, max([history.history['loss'][0], history.history['val_loss'][0]]))
		plt.legend(loc='upper right')

		plt.savefig(DIR+HISTORY_IMAGE_FILE, dpi=100)
		if stdout == True:
			plt.show()


	def test_model(self, data):
		X_train, Y_train, X_test, Y_test = data

		model_file = os.path.join(DIR, MODEL_FILE)
		weight_file = os.path.join(DIR, WEIGHT_FILE)

		with open(model_file, 'r') as fp:
		    model = model_from_json(fp.read())
		model.load_weights(weight_file)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		
		if not DATA_AUGMENTATION:
			loss, acc = model.evaluate(X_test, Y_test, verbose=0)
		else:
			# Make a generator for test data
			test_datagen = ImageDataGenerator(zca_whitening=True)
			test_datagen.fit(X_test)
			test_generator = test_datagen.flow(X_test, Y_test)

			loss, acc = model.evaluate_generator(test_generator, val_samples=X_test.shape[0])

		print('Test loss: %s, Test acc: %s' % (loss, acc))
		print('')


if __name__ == "__main__":
	Test().main()