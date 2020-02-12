# Step 1: import the Torchreid library
import torchreid


"""
Epoch: [60/60][350/404] Time 0.190 (0.203)      Data 0.000 (0.014)      Loss 1.0596 (1.0721)    Acc 100.00 (99.99)      Lr 0.000003     eta 0:00:10
Epoch: [60/60][360/404] Time 0.189 (0.203)      Data 0.000 (0.014)      Loss 1.0595 (1.0721)    Acc 100.00 (99.99)      Lr 0.000003     eta 0:00:08
Epoch: [60/60][370/404] Time 0.187 (0.203)      Data 0.000 (0.013)      Loss 1.1000 (1.0721)    Acc 100.00 (99.99)      Lr 0.000003     eta 0:00:06
Epoch: [60/60][380/404] Time 0.188 (0.202)      Data 0.000 (0.013)      Loss 1.0709 (1.0722)    Acc 100.00 (99.99)      Lr 0.000003     eta 0:00:04
Epoch: [60/60][390/404] Time 0.196 (0.202)      Data 0.001 (0.013)      Loss 1.0650 (1.0722)    Acc 100.00 (99.98)      Lr 0.000003     eta 0:00:02
Epoch: [60/60][400/404] Time 0.194 (0.202)      Data 0.000 (0.012)      Loss 1.0691 (1.0722)    Acc 100.00 (99.98)      Lr 0.000003     eta 0:00:00
=> Final test
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-2048 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-2048 matrix
Speed: 0.0236 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 68.4%
CMC curve
Rank-1  : 85.2%
Rank-5  : 93.6%
Rank-10 : 95.9%
Rank-20 : 97.3%
Checkpoint saved to "log/resnet50\model.pth.tar-60"
Elapsed 1:45:48
"""

def main():
	#train()
	#test()
	vis_rank()


def train():
	# Step 2: construct data manager
	datamanager = torchreid.data.ImageDataManager(
		root='D:/workspace/datasets',
		sources='market1501',
		targets='market1501',
		height=256,
		width=128,
		batch_size_train=32,
		batch_size_test=100,
		transforms=['random_flip', 'random_crop']
	)

	# Step 3: construct CNN model
	model = torchreid.models.build_model(
		name='resnet50',
		num_classes=datamanager.num_train_pids,
		loss='softmax',
		pretrained=True
	)
	model = model.cuda()
	
	# Step 4: initialise optimiser and learning rate scheduler
	optimizer = torchreid.optim.build_optimizer(
		model,
		optim='adam',
		lr=0.0003
	)
	scheduler = torchreid.optim.build_lr_scheduler(
		optimizer,
		lr_scheduler='single_step',
		stepsize=20
	)

	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer,
		scheduler=scheduler,
		label_smooth=True
	)

	# Step 6: run model training and test
	engine.run(
		#save_dir='log/resnet50',
		save_dir='../experiments/models/checkpoints',
		max_epoch=60,
		eval_freq=10,
		print_freq=10,
		test_only=False
	)

def test():
	# Step 2: construct data manager
	datamanager = torchreid.data.ImageDataManager(
		root='D:/workspace/datasets',
		sources='market1501',
		targets='market1501',
		height=256,
		width=128,
		batch_size_train=32,
		batch_size_test=100,
		transforms=['random_flip', 'random_crop']
	)

	# Step 3: construct CNN model
	model = torchreid.models.build_model(
		name='resnet50',
		num_classes=datamanager.num_train_pids,
		loss='softmax',
		pretrained=True
	)
	model = model.cuda()
	
	# Step 4: initialise optimiser and learning rate scheduler
	optimizer = torchreid.optim.build_optimizer(
		model,
		optim='adam',
		lr=0.0003
	)
	scheduler = torchreid.optim.build_lr_scheduler(
		optimizer,
		lr_scheduler='single_step',
		stepsize=20
	)

	weight_path = '../experiments/models/model_market1501_resnet50.pth.tar-60'
	torchreid.utils.load_pretrained_weights(model, weight_path)

	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer,
		scheduler=scheduler,
		label_smooth=True
	)

	# Step 6: run model training and test
	engine.run(
		#save_dir='log/resnet50',
		save_dir='../experiments/models/checkpoints',
		max_epoch=60,
		eval_freq=10,
		print_freq=10,
		test_only=True
	)


def vis_rank():
	# Step 2: construct data manager
	datamanager = torchreid.data.ImageDataManager(
		root='D:/workspace/datasets',
		sources='market1501',
		targets='market1501',
		height=256,
		width=128,
		batch_size_train=32,
		batch_size_test=100,
		transforms=['random_flip', 'random_crop']
	)

	# Step 3: construct CNN model
	model = torchreid.models.build_model(
		name='resnet50',
		num_classes=datamanager.num_train_pids,
		loss='softmax',
		pretrained=True
	)
	model = model.cuda()
	
	# Step 4: initialise optimiser and learning rate scheduler
	optimizer = torchreid.optim.build_optimizer(
		model,
		optim='adam',
		lr=0.0003
	)
	scheduler = torchreid.optim.build_lr_scheduler(
		optimizer,
		lr_scheduler='single_step',
		stepsize=20
	)

	weight_path = '../experiments/models/model_market1501_resnet50.pth.tar-60'
	torchreid.utils.load_pretrained_weights(model, weight_path)

	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer,
		scheduler=scheduler,
		label_smooth=True
	)

	# Step 6: run model training and test
	engine.run(
		#save_dir='log/resnet50',
		save_dir='../experiments/',
		max_epoch=60,
		eval_freq=10,
		print_freq=10,
		test_only=True,
		visrank=True
	)

if __name__ == "__main__":
	main()

