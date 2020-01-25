# Visualize Object Detection Annotation data.
This code visualize annotation data for object detection using OpenCV.


# Feature
- General process to draw annotation data for object detection
- Automatic selection of color per object using colormap


# Requirement


# Requirement
- python 3.6.8
- pytorch 1.1.0
- torchvision 0.3.0
- scikit-learn 0.20.3
- cuda 9.0
- cudnn 7.1 


# Installation
 
```sh
git clone https://github.com/iShoto/testpy.git
```


# Usage
 
```sh
$ cd testpy/codes/20200105_metric_learning_mnist_query_and_gallery/src/
$ python python image_seach_query.py
num query: 1, num gallery: 100

Query Image Label: 9

Search Result
        dist                      img_path  label
0   0.005382  ../inputs/gallery/9_1801.png      9
1   0.018921  ../inputs/gallery/9_4237.png      9
2   0.036690   ../inputs/gallery/9_481.png      9
3   0.047976  ../inputs/gallery/9_7380.png      9
4   0.069177  ../inputs/gallery/9_8213.png      9
5   0.076138  ../inputs/gallery/9_3970.png      9
6   0.078646  ../inputs/gallery/9_2685.png      9
7   0.107746  ../inputs/gallery/9_5977.png      9
8   0.387746  ../inputs/gallery/9_4505.png      9
9   0.523175  ../inputs/gallery/3_8981.png      3
10  0.538863   ../inputs/gallery/3_927.png      3
11  0.560314   ../inputs/gallery/3_142.png      3
12  0.565455  ../inputs/gallery/3_8451.png      3
13  0.582634  ../inputs/gallery/3_4755.png      3
14  0.586750  ../inputs/gallery/3_2174.png      3
15  0.589938  ../inputs/gallery/3_9986.png      3
16  0.675965  ../inputs/gallery/1_4491.png      1
17  0.682165  ../inputs/gallery/3_8508.png      3
18  0.683414  ../inputs/gallery/3_4785.png      3
19  0.698637  ../inputs/gallery/1_1038.png      1
```


# Author
Shoto I.


# License
[MIT license](https://github.com/iShoto/testpy/blob/master/LICENSE).
