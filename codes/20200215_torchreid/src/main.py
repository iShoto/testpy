# Step 1: import the Torchreid library
import torchreid


def main():
	# Training
	datamanager, model, optimizer, scheduler = get_items()
	#train(datamanager, model, optimizer, scheduler)

	# Test
	weight_path = '../experiments/models/model_market1501_resnet50.pth.tar-60'
	torchreid.utils.load_pretrained_weights(model, weight_path)
	test(datamanager, model, optimizer)
	vis_rank(datamanager, model, optimizer)


def get_items():
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

	return datamanager, model, optimizer, scheduler


def train(datamanager, model, optimizer, scheduler):
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
		save_dir='../experiments/models/checkpoints',
		max_epoch=60,
		eval_freq=10,
		print_freq=10,
		test_only=False
	)


def test(datamanager, model, optimizer):
	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer
	)

	# Step 6: run model training and test
	engine.run(
		#save_dir='../experiments/models/checkpoints',
		test_only=True
	)


def vis_rank(datamanager, model, optimizer):
	#torchreid.utils.load_pretrained_weights(model, weight_path)

	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer
	)

	# Step 6: run model training and test
	engine.run(
		save_dir='../experiments/',
		test_only=True,
		visrank=True
	)


if __name__ == "__main__":
	main()

