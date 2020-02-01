# Modification of PyTorch Object Detection Tutorial
Modified PyTorch Object Detection Tutorial.


# Feature
- Save checkpoints
- Calculate mAP with Pascal VOC method
- Draw ground truth and detection results on images


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
$ cd testpy/codes/20200202_pytorch_object_detection/src/
$ python train.py
Loading a model...
Epoch: [0]  [ 0/60]  eta: 0:05:40  lr: 0.000090  loss: 0.9656 (0.9656)  loss_classifier: 0.7472 (0.7472)  loss_box_reg: 0.1980 (0.1980)  loss_objectness: 0.0093 (0.0093)  loss_rpn_box_reg: 0.0111 (0.0111)  time: 5.6684  data: 3.8234  max mem: 3548
Epoch: [0]  [10/60]  eta: 0:00:47  lr: 0.000936  loss: 0.7725 (0.6863)  loss_classifier: 0.4681 (0.4597)  loss_box_reg: 0.1726 (0.2008)  loss_objectness: 0.0127 (0.0163)  loss_rpn_box_reg: 0.0070 (0.0095)  time: 0.9561  data: 0.3484  max mem: 4261
Epoch: [0]  [20/60]  eta: 0:00:28  lr: 0.001783  loss: 0.4149 (0.5500)  loss_classifier: 0.2203 (0.3413)  loss_box_reg: 0.1567 (0.1806)  loss_objectness: 0.0127 (0.0188)  loss_rpn_box_reg: 0.0060 (0.0093)  time: 0.4775  data: 0.0011  max mem: 4261
Epoch: [0]  [30/60]  eta: 0:00:19  lr: 0.002629  loss: 0.3370 (0.4652)  loss_classifier: 0.1338 (0.2607)  loss_box_reg: 0.1504 (0.1787)  loss_objectness: 0.0094 (0.0152)  loss_rpn_box_reg: 0.0084 (0.0106)  time: 0.4745  data: 0.0011  max mem: 4261
Epoch: [0]  [40/60]  eta: 0:00:12  lr: 0.003476  loss: 0.2195 (0.3939)  loss_classifier: 0.0534 (0.2088)  loss_box_reg: 0.1345 (0.1620)  loss_objectness: 0.0063 (0.0129)  loss_rpn_box_reg: 0.0094 (0.0102)  time: 0.4814  data: 0.0011  max mem: 4458
Epoch: [0]  [50/60]  eta: 0:00:05  lr: 0.004323  loss: 0.1478 (0.3427)  loss_classifier: 0.0442 (0.1765)  loss_box_reg: 0.0842 (0.1432)  loss_objectness: 0.0021 (0.0118)  loss_rpn_box_reg: 0.0091 (0.0112)  time: 0.5083  data: 0.0014  max mem: 4623
Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.1176 (0.3073)  loss_classifier: 0.0433 (0.1562)  loss_box_reg: 0.0561 (0.1293)  loss_objectness: 0.0009 (0.0102)  loss_rpn_box_reg: 0.0117 (0.0117)  time: 0.5165  data: 0.0013  max mem: 4623
Epoch: [0] Total time: 0:00:34 (0.5814 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:03:14  model_time: 0.1120 (0.1120)  evaluator_time: 0.0010 (0.0010)  time: 3.8935  data: 3.7785  max mem: 4623
Test:  [49/50]  eta: 0:00:00  model_time: 0.0900 (0.0962)  evaluator_time: 0.0010 (0.0012)  time: 0.0994  data: 0.0006  max mem: 4623
Test: Total time: 0:00:08 (0.1791 s / it)
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.641
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.987
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.859
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.700
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706
Saved a model checkpoint at ../experiments/models/checkpoints/PennFudanPed_FasterRCNN-ResNet50_epoch=1.pth
...

$ python .\test.py
Loading a model from ../experiments/models/PennFudanPed_FasterRCNN-ResNet50_epoch=10.pth
Detecting objects... 100%
========================== DETECTION RESULTS ==========================
   label  score  xmin  ymin  xmax  ymax                                         image_path
0      1  0.999   294   129   447   419  D:/workspace/datasets/PennFudanPed/PNGImages/F...
1      1  0.999   361   135   456   399  D:/workspace/datasets/PennFudanPed/PNGImages/F...
2      1  0.999   207   100   350   382  D:/workspace/datasets/PennFudanPed/PNGImages/F...
3      1  0.999     0   111    88   383  D:/workspace/datasets/PennFudanPed/PNGImages/F...
4      1  0.999    37   100    97   362  D:/workspace/datasets/PennFudanPed/PNGImages/F...
5      1  0.999    40   106    87   268  D:/workspace/datasets/PennFudanPed/PNGImages/F...
6      1  0.998   268    92   397   374  D:/workspace/datasets/PennFudanPed/PNGImages/F...
7      1  0.998   260   191   294   345  D:/workspace/datasets/PennFudanPed/PNGImages/F...
8      1  0.998   262    97   338   357  D:/workspace/datasets/PennFudanPed/PNGImages/F...
9      1  0.999   384   192   551   482  D:/workspace/datasets/PennFudanPed/PNGImages/F...
Detection results saved to ../experiments/results/tables/dets.csv
Making gt text files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 746.27it/s]
Making det text files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 724.33it/s]
65.53% = 1 AP
mAP = 65.53%
  class_name        ap  recall  precision   gt  n_det   tp  fp
0          1  0.655343     1.0   0.581395  125    215  125  90
Score saved to ../experiments/results/tables/score.csv
Drawing gt and dets: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 62.81it/s]
```


# Author
Shoto I.


# License
[MIT license](https://github.com/iShoto/testpy/blob/master/LICENSE).
