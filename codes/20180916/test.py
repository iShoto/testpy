# coding: utf-8

from scipy.misc import imread
import matplotlib.pyplot as plt

img = imread('IMG_1382.JPG')
#fig = plt.figure(figsize=(10,10), dpi=200)
plt.imshow(img / 255.)
plt.axis('off')
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.savefig('IMG_1382_ex.JPG', bbox_inches='tight', pad_inches=0)