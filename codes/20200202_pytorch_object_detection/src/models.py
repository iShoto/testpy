import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_resnet50(num_classes):
	# load an instance segmentation model pre-trained pre-trained on COCO
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	# now get the number of input features for the mask classifier
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	# and replace the mask predictor with a new one
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
													   hidden_layer,
													   num_classes)

	return model


def get_fasterrcnn_resnet50(num_classes, pretrained=False):
	# load a model pre-trained pre-trained on COCO
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

	# replace the classifier with a new one, that has num_classes which is user-defined
	#num_classes = 2  # 1 class (person) + background
	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model