import cv2
import numpy as np

def main():
	#print('Hello test.py!')
	back_img_path = '../data/inputs/0000_frame_000000000925.png'
	back_img = cv2.imread(back_img_path, 1)

	img_path = '../data/inputs/shoto_tech_logo.png'
	img = cv2.imread(img_path, 1)
	h, w, c = img.shape
	scale = 0.49
	img = cv2.resize(img, (int(w*scale), int(h*scale)))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype('uint8')
	#img = cv2.GaussianBlur(img, (9, 9), 0)
	
	img_inv = cv2.bitwise_not(img)
	cv2.imwrite('../data/inputs/shoto_tech_logo_inv.png', img_inv)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
	img_inv_bold_blur = cv2.erode(img_inv, kernel)
	#img_inv_bold_blur = cv2.GaussianBlur(img_inv_bold, (3, 3), 0)
	
	cv2.imwrite('../data/inputs/shoto_tech_logo_inv_bold_blur.png', img_inv_bold_blur)
	
	img_logo = np.where(img>=245, img, img_inv_bold_blur)
	cv2.imwrite('../data/inputs/img_logo.png', img_logo)

	back_h, back_w, back_c = back_img.shape
	img_logo = cv2.cvtColor(img_logo, cv2.COLOR_GRAY2BGR)
	fore_h, fore_w, fore_c = img_logo.shape	

	ymin = (back_h//2) - (fore_h//2)
	ymax = (back_h//2) + (fore_h//2)
	xmin = (back_w//2) - (fore_w//2)
	xmax = (back_w//2) + (fore_w//2)
	print(ymin, ymax, xmin, xmax)

	print(img_logo)

	back_crop_img = back_img[ymin:ymax, xmin:xmax]

	#back_img[ymin:ymax, xmin:xmax] = img_logo
	back_crop_img = np.where(img_logo<250, img_logo, back_crop_img)
	cv2.imwrite('../data/inputs/back_crop.png', back_crop_img)

	back_img[ymin:ymax, xmin:xmax] = back_crop_img
	cv2.imwrite('../data/inputs/credit.png', back_img)

	1/0
	
	img = np.where(img>=127, 255, 0)
	img = img.astype('uint8')

	#img = cv2.GaussianBlur(img, (5, 5), 0)
	#cv2.imwrite('../data/inputs/temp_0.png', img)

	#Apply dilate function on input image. Dilation will increase brightness, First Parameter is the original image, 
	#second is the dilated image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	img_bold = cv2.erode(img, kernel)
	#img_bold = cv2.blur(img_bold, (11, 11))
	img_bold = cv2.GaussianBlur(img_bold, (15, 15), 0)
	
	cv2.imwrite('../data/inputs/temp_1.png', img_bold)

	img = cv2.bitwise_not(img)
	img = cv2.GaussianBlur(img, (7, 7), 0)
	cv2.imwrite('../data/inputs/temp_0.png', img)

	#img_logo = cv2.absdiff(img_bold, img)
	img_logo = np.where(img==0, img, img_bold)
	cv2.imwrite('../data/inputs/temp_2.png', img_logo)

	back_h, back_w, back_c = back_img.shape
	print(back_h, back_w, back_c)

	img_logo = cv2.cvtColor(img_logo, cv2.COLOR_GRAY2BGR)
	fore_h, fore_w, fore_c = img_logo.shape
	print(fore_h, fore_w, fore_c)

	#1080 1920 3
	#209 896 3

	ymin = (back_h//2) - (fore_h//2)
	ymax = (back_h//2) + (fore_h//2)
	xmin = (back_w//2) - (fore_w//2)
	xmax = (back_w//2) + (fore_w//2)
	print(ymin, ymax, xmin, xmax)

	back_img[ymin:ymax, xmin:xmax] = img_logo
	#back_img[ymin:fore_h, xmin:fore_w] = img_logo

	cv2.imwrite('../data/inputs/credit.png', back_img)

	#img = cv2.bitwise_and(img_logo, back_img)

	1/0
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

