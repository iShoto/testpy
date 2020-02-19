import cv2
import numpy as np

def main():
	#print('Hello test.py!')
	img_path = '../data/inputs/sho_original_logo.png'
	img = cv2.imread(img_path, 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype('uint8')
	
	h, w = img.shape
	#scale = 0.2
	#img = cv2.resize(img, (int(w*scale), int(h*scale)))
	img = np.where(img>=127, 255, 0)
	#kernel = np.ones((5,5), np.uint8)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	img = cv2.dilate(img)
	#img = cv2.erode(img, kernel=3)

	cv2.imwrite('../data/inputs/temp_1.png', img)


if __name__ == "__main__":
	main()

