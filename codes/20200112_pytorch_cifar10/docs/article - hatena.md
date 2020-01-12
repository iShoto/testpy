CIFAR10の画像分類は
[PyTorchのチュートリアル](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
に従ったらできるようになったのだが、
オリジナルモデルだったためResNet18に変更しようとしたら少しつまづいた。
再度つまづかないために、ここに実行手順をコード解説付きでまとめておく。
なお全コードは
[ここ](https://github.com/iShoto/testpy/tree/master/codes/20200112_pytorch_cifar10)
に置いてある。


## 概要

実行手順は次の通り。

1. データ取得
2. モデル定義
3. 損失関数と最適化関数の定義
4. 学習と検証

これらは`main()`で次のように実行される。

```python
def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load dataset.
	train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
	
	# Set a model.
	model = get_model(args.model_name)
	model = model.to(device)

	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
		test_acc, test_loss = test(model, device, test_loader, criterion)
		
		# Output score.
		stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))

		# Save a model checkpoint.
		model_ckpt_path = args.model_ckpt_path_temp.format(args.dataset_name, args.model_name, epoch+1)
		torch.save(model.state_dict(), model_ckpt_path)
		print('Saved a model checkpoint at {}'.format(model_ckpt_path))
		print('')
```


## 1. データ取得

CIFAR10を利用する。
今後の拡張性も考えて、データセット読み込み用の`datasets`ディレクトリーを作って、
CIFAR10関連のコードは`cifar10.py`にまとめている。

```python
from datasets import cifar10

def main():
	...

	# Load dataset.
	train_loader, test_loader, class_names = cifar10.load_data(args.data_dir)
```

`cifar10.py`の中身は次の通り。
CIFAR10は`torchvision`があればOKなので実装は簡単。
trainとtestのDataLoaderとクラス名を返す。

```python
import torchvision
import torchvision.transforms as transforms

def load_data(data_dir):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
	test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
	class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return train_loader, test_loader, class_names
```


## 2. モデル定義

[Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)の`models`に
ResNet以外にも様々なコードがあったのでディレクトリーごと拝借した。


```python
from models import *

def main():
	...

	# Set a model.
	model = get_model(args.model_name)
```

色々あるので引数にモデル名を入れれば取得できるようにした。

```python
def get_model(model_name):
	if model_name == 'VGG19':
		model = VGG('VGG19')
	elif model_name == 'ResNet18':
		model = ResNet18()
	elif model_name == 'PreActResNet18':
		model = PreActResNet18()
	...
	elif model_name == 'SENet18':
		model = SENet18()
	elif model_name == 'ShuffleNetV2':
		model = ShuffleNetV2(1)
	elif model_name == 'EfficientNetB0':
		model = EfficientNetB0()
	else:
		print('{} does NOT exist in repertory.'.format(model_name))
		sys.exit(1)
```

## 3. 損失関数と最適化関数の定義

今回はオーソドックスにCross EntropyとSGDを各々セット。

```python
def main():
	...
	
	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
```


## 4. 学習と検証

これまでセットして変数を`train()`に入力して訓練を開始する。

```python
def main():
	...

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
```

`train()`では一般的な処理を踏む。
スコア算出のために、出力結果と正解のリスト、およびロスを貯める。

```python
def train(model, device, train_loader, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()

		# Calculate score at present.
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
		if batch_idx % 10 == 0 and batch_idx != 0:
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

	# Calculate score.
	train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

	return train_acc, train_loss
```

スコア算出は`calc_score()`で行う。
精度の算出はscikit-learnの`classification_report()`を用いる。

```python
def calc_score(output_list, target_list, running_loss, data_loader):
	# Calculate accuracy.
	result = classification_report(output_list, target_list, output_dict=True)
	acc = round(result['weighted avg']['f1-score'], 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss
```

検証用のメソッド`test()`も中身は学習する以外はだいたい`train()`と同じ。
両メソッドから算出したスコアを取得して表示する。

```python
def main():
	...

	# Train and test.
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
		test_acc, test_loss = test(model, device, test_loader, criterion)
		
		# Output score.
		stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
```

以下が実行中の標準出力。
10バッチごとと1エポックごとに出力する。

```bash
$ python train.py
Files already downloaded and verified
Files already downloaded and verified
batch:  10/391, train acc: 0.176831, train loss: 0.000512
batch:  20/391, train acc: 0.208884, train loss: 0.000931
batch:  30/391, train acc: 0.214069, train loss: 0.001356
...
batch: 390/391, train acc: 0.424302, train loss: 0.012254
epoch:   1, train acc: 0.424302, train loss: 0.012254, test acc: 0.539407, test loss: 0.012638
Saved a model checkpoint at ../experiments/models/checkpoints/CIFAR10_ResNet18_epoch=1.pth
```

ほぼ`test()`と同じで、学習済みモデルを読み込んで評価を行う`test.py`も作成した。
モデルは10エポックで精度77.9%ほどとなることを確認。

```bash
$ python test.py
Files already downloaded and verified
Files already downloaded and verified
Loaded a model from ../experiments/models/CIFAR10_ResNet18_epoch=10.pth
test acc: 0.779172, test loss: 0.006796
```


## 参考文献
- [Train CIFAR10 with PyTorch - github/kuangliu](https://github.com/kuangliu/pytorch-cifar)