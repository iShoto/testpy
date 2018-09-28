# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange

os.makedirs('./images/', exist_ok=True)

i = 0
for i in trange(10000, desc='saving images'):
	img = np.full((64, 64, 3), 128)
	plt.imshow(img / 255.)
	plt.axis('off')
	plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
	plt.savefig('./images/%s.jpg'%str(i).zfill(4), bbox_inches='tight', pad_inches=0)
	plt.close()
