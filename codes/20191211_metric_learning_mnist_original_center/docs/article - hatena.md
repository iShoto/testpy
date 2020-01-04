Person ReIDが必要になったので、まずはMNISTを題材に距離学習を勉強している。
あと、これまでKerasを使ってきたけど、PyTorch使えないと厳しい世の中になってきたので、
PyTorchについて色々調べつつ実装してみた。

なお今回は[こちらの記事](https://cpp-learning.com/center-loss/)（以下、参照記事）を参考にしている。
距離学習をメインで学びたい人は本記事より参照記事を読むことをお薦めする。
本記事はPyTorch入門みたいな要素が強いので。


## 概要
距離学習をもの凄く簡単に言うと画像分類の拡張。
なので、処理フローはだいたい画像分類と同じで以下のようになる。

1. データ準備
2. モデル定義
3. 損失関数定義
4. 最適化関数定義
5. 訓練検証

距離学習は、同じクラスは近く異なるクラスは遠くなるようにモデルを学習することで、
未知のクラスの同定を行えるのが画像分類と違うところ。
ポイントは損失関数で、今回はCenterLossというのを使っているが、
説明は[参照記事](https://cpp-learning.com/center-loss/)が詳しい。

本記事で説明するコードは[ここ](https://github.com/iShoto/testpy/tree/master/codes/20200104_metric_learning_mnist)にある。
以下の`train_mnist_original_center.py`の`main()`を実行すると、
参照記事と同じような結果が得られるが、個人的にコード整理してみたので、
上述の処理フローに従って順に説明する。


```python
def main():
	# Arguments
	args = parse_args()

	# Dataset
	train_loader, test_loader, classes = mnist_loader.load_dataset(args.dataset_dir, img_show=True)

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Model
	model = Net().to(device)
	print(model)

	# Loss
	nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
	loss_weight = 1
	centerloss = CenterLoss(10, 2).to(device)
	
	# Optimizer
	dnn_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
	sheduler = lr_scheduler.StepLR(dnn_optimizer, 20, gamma=0.8)
	center_optimizer = optim.SGD(centerloss.parameters(), lr =0.5)
	
	print('Start training...')
	for epoch in range(100):
		# Update parameters.
		epoch += 1
		sheduler.step()

		# Train and test a model.
		train_acc, train_loss, feat, labels = train(device, train_loader, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer)
		test_acc, test_loss = test(device, test_loader, model, nllloss, loss_weight, centerloss)
		stdout_temp = 'Epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
		print(stdout_temp.format(epoch, train_acc, train_loss, test_acc, test_loss))
		
		# Visualize features of each class.
		vis_img_path = args.vis_img_path_temp.format(str(epoch).zfill(3))
		visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, vis_img_path)

		# Save a trained model.
		model_path = args.model_path_temp.format(str(epoch).zfill(3))
		torch.save(model.state_dict(), model_path)
```


## 1. データ準備
先に引数の説明を少し。

```python
	# Arguments
	args = parse_args()
```

`dataset_dir`はMNISTデータの保存場所。
後述するPyTorchの機能でここにダウンロードしてくれる。
`model_path_temp`は学習済みモデルのチェックポイント。
各エポック終了後に保存する。
`vis_img_path_temp`はMNISTの各クラスの特徴分布を可視化した画像。
こちらも各エポック終了後に保存する。
だんだんとクラス内でまとまりクラス間が離れていく様子が確認できる。
下図は100エポック後の特徴分布。

```python
def parse_args():
	arg_parser = argparse.ArgumentParser(description="parser for focus one")

	arg_parser.add_argument("--dataset_dir", type=str, default='../inputs/')
	arg_parser.add_argument("--model_path_temp", type=str, default='../outputs/models/checkpoints/mnist_original_softmax_center_epoch_{}.pth')
	arg_parser.add_argument("--vis_img_path_temp", type=str, default='../outputs/visual/epoch_{}.png')
	
	args = arg_parser.parse_args()

	return args
```

[f:id:Shoto:20200104213047p:plain]


では、MNISTのデータセットを取得する。
MNIST関連は、`mnist_loader.py`という別ファイルを作って処理している。

```python
# Dataset
train_loader, test_loader, classes = mnist_loader.load_dataset(args.dataset_dir, img_show=True)
```

`load_dataset`は、train_loader、test_loader、クラス名を取得するメソッド。
ここからPyTorch色が強くなるが、データ準備では次の手順を踏む。

#### 1. 画像の前処理
`torchvision`の`transform`を利用する。
`ToTensor()`でPyTorchの`torch.Tensor`型に変換する。
他にも、クロップやフリップなどData Augmentation的な事を行えるが、今回は未実施。
今回は`Normalize()`で正規化を行っている。
なおMNISTは自然画像ではないので、平均0.1307、標準偏差0.3081となるようにする。

```python
from torchvision import transforms
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])
```

#### 2. 画像データセットを取得
`torchvision`の`datasets.MNIST`を使うとMNISTが簡単に利用できる。
第1引数はMNISTデータの保存場所。
第2引数でtrain用かtest用かを選ぶ。
第3引数がTrueの場合は保存場所にMNISTデータがない場合に自動でダウンロードしてくれる。
第4引数で先に定義したtransformをセットする。

```python
from torchvision import datasets
trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
```

#### 3. データローダーを定義
`torch.utils.data`の`DataLoader`を利用して、指定バッチ数分のデータを取得する。
第1引数は2で定義したデータセット。
第2引数はバッチサイズ。
第3引数はデータシャッフルするか否か。訓練時はTrueが妥当。
第4引数はデータロードの並列処理数。

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
```

上記のメソッドを組み合わせることで、`mnist_loader.load_dataset()`は次のようになる。

```python
def load_dataset(dataset_dir, train_batch_size=128, test_batch_size=128, img_show=False):
	# Dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	trainset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
	train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	if img_show == True:
		show_data(train_loader)

	return train_loader, test_loader, classes
```

`show_data()`はMNISTを可視化するメソッド。
`torchvision.utils.make_grid()`により train_loader のバッチを簡単に可視化できる。

```python
def show_data(data_loader):
	images, labels = iter(data_loader).next()  # data_loader のミニバッチの image を取得
	img = torchvision.utils.make_grid(images, nrow=16, padding=1)  # nrom*nrom のタイル形状の画像を作る
	plt.ion()
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 画像を matplotlib 用に変換
	plt.draw()
	plt.pause(3)  # Display an image for three seconds.
	plt.close()
```


## 2. モデル定義
PyTorchでは処理をGPUとCPUのどちらで行うか`torch.device`で明示的に選択して、
それをモデルやデータにセットする必要がある。
モデル定義はMNIST向けのを`mnist_net.py`の`Net()`で別途定義している。

```python
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Net().to(device)
print(model)
```

`mnist_net.py`の`Net()`は次の通り。
**Define by Run**では、
`__init__()`で計算グラフを幾つか定義して、ネットワーク生成時に１度だけ呼びし、
データ入力時に`forward()`を呼び出す使用となっている。
6つの畳み込み層とPReLUの後、2次元空間に落とし込んだ特徴ip1と、
それをPReLUに通して10次元空間に写像したip2を出力する。
ip1が特徴分布で、ip2はクラス分類に利用する。

```python
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
		self.prelu1_1 = nn.PReLU()
		self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
		self.prelu1_2 = nn.PReLU()
		self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.prelu2_1 = nn.PReLU()
		self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
		self.prelu2_2 = nn.PReLU()
		self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
		self.prelu3_1 = nn.PReLU()
		self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
		self.prelu3_2 = nn.PReLU()
		self.ip1 = nn.Linear(128*3*3, 2)
		self.preluip1 = nn.PReLU()
		self.ip2 = nn.Linear(2, 10, bias=False)
 
	def forward(self, x):
		x = self.prelu1_1(self.conv1_1(x))
		x = self.prelu1_2(self.conv1_2(x))
		x = F.max_pool2d(x,2)
		x = self.prelu2_1(self.conv2_1(x))
		x = self.prelu2_2(self.conv2_2(x))
		x = F.max_pool2d(x,2)
		x = self.prelu3_1(self.conv3_1(x))
		x = self.prelu3_2(self.conv3_2(x))
		x = F.max_pool2d(x,2)
		x = x.view(-1, 128*3*3)
		ip1 = self.preluip1(self.ip1(x))
		ip2 = self.ip2(ip1)
		return ip1, F.log_softmax(ip2, dim=1)
```

なお`nn.PReLU`はReLUの改良の改良。
LeakyReLU で `x < 0` の時に `y < 0` することで学習が進みやすくなったものの、
パラメーター α が増えたため、それを減らすべく学習させることにしたのがPReLU。
ちなみにPReLUは、"a Parametric Rectified Linear Unit" の略。
まとめると以下のようになる。

```
ReLU
	y = x (0 =< x)
	y = 0 (x < 0)

LeakyReLU
	y = x (0 =< x)
	y = αx (x < 0), set α as a parameter

PReLU
	y = x (0 =< x)
	y = αx (x < 0), learning α
```

また`view`は`numyp.reshape`と同じ。
第一引数が-1のとき、第二引数の形に自動調整してくれる。
上記の場合だと、x の shape が (3, 3, 128) になるので、
1次元に変換している。


## 3. 損失関数定義
損失関数は、クラス分類用の`NLL Loss`にMetric Learninig用の`Center Loss`を加重加算したものを利用する。

```
Loss = NLL Loss + α * Center Loss, α is weight
```

`NLL Loss`はNegative Log-Likelihood (NLL) Lossの略。
softmaxの最大値は結果の確信度を表すが、それをマイナスの対数で取った値となる。
NLL Lossにより、高い確信度であれば低いロス、低い確信度であれば高いロスを割り当てることができる。
ip2のsoftmax（定義したモデルの出力）を入力とする。

一方`Center Loss`は特徴の中心の損失関数。ip1を入力する。
詳しい説明は[参照記事](https://cpp-learning.com/center-loss/)に任せる。
ちなみに、自分は距離学習に
[ArcFace](https://arxiv.org/abs/1801.07698)から入ったので、
`Center Loss`はこの記事以外では使わないかな、と思っている。

PyTorchで、損失関数は次のように定義される。
`CenterLoss()`は自作関数でクラス数と特徴数が引数となる。

```python
# NLL Loss & Center Loss
nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
loss_weight = 1  # weight
centerloss = CenterLoss(10, 2).to(device)
# Loss
loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
```

## 4. 最適化関数定義
最適化関数にはSGDを利用するが、クラス分類と距離特徴の両方を行っているので、
それぞれで最適化関数を定義する。
前者については、学習率の減衰を`lr_scheduler.StepLR()`で行う。
第一引数はクラス分類用の最適化関数、第二引数は学習率を更新するタイミングのエポック数、
第三引数は学習率の更新率。

```python
# Optimizer
dnn_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
center_optimizer = optim.SGD(centerloss.parameters(), lr =0.5)
import torch.optim.lr_scheduler as lr_scheduler
sheduler = lr_scheduler.StepLR(dnn_optimizer, 20, gamma=0.8)
```

## 5. 訓練検証
これまで定義してきた変数と関数を利用して訓練を行う。
エポックごとに`train()`を呼び出す。

```python
train_acc, train_loss, feat, labels = train(device, train_loader, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer)
```

`train()`は一般的な機械学習。

```python
def train(device, train_loader, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer):
	running_loss = 0.0
	pred_list = []
	label_list = []
	ip1_loader = []
	idx_loader = []
	
	model.train()
	for i,(imgs, labels) in enumerate(train_loader):
		# Set batch data.
		imgs, labels = imgs.to(device), labels.to(device)
		# Predict labels.
		ip1, pred = model(imgs)
		# Calculate loss.
		loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
		# Initilize gradient.
		dnn_optimizer.zero_grad()
		center_optimizer.zero_grad()
		# Calculate gradient.
		loss.backward()
		# Update parameters.
		dnn_optimizer.step()
		center_optimizer.step()
		# For calculation.
		running_loss += loss.item()
		pred_list += [int(p.argmax()) for p in pred]
		label_list += [int(l) for l in labels]
		# For visualization.
		ip1_loader.append(ip1)
		idx_loader.append((labels))
	
	# Calculate training accurary and loss.
	result = classification_report(pred_list, label_list, output_dict=True)
	train_acc = round(result['weighted avg']['f1-score'], 6)
	train_loss = round(running_loss / len(train_loader.dataset), 6)
	
	# Concatinate features and labels.
	feat = torch.cat(ip1_loader, 0)
	labels = torch.cat(idx_loader, 0)
	
	return train_acc, train_loss, feat, labels
```

PyTorchでは訓練中、lossとoptimizerはバッチごとに次の手順を踏んで、パラメーターを更新していく。

```
optimizer.zero_grad()  # 勾配の初期化
loss.backward()  # 勾配の計算
optimizer.step()  # パラメータの更新
```

また`sklearn.metrics`の`classification_report()`を利用すると、
簡単に精度が算出できる。

訓練と検証の精度と損失の変化は以下の通り。

```

```



## 参考文献
- [PyTorch まずMLPを使ってみる - cedro-blog](http://cedro3.com/ai/pytorch-mlp/)
- [Normalization in the mnist example - PyTorch Forums](https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457)
- [ChainerのDefine by Runとは？ - HELLO CYBERNETICS](https://www.hellocybernetics.tech/entry/2017/01/14/060758)
- [LeakyRelu活性化関数 - Thoth Children](http://www.thothchildren.com/chapter/59b93f7575704408bd4300f2)
- [PRelu活性化関数 - Thoth Children](http://www.thothchildren.com/chapter/59b940a475704408bd4300f8)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification - arXiv](https://arxiv.org/abs/1502.01852)
- [Understanding softmax and the negative log-likelihood - Lj Miranda](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)
- [実践Pytorch - Qiita](https://qiita.com/perrying/items/857df46bb6cdc3047bd8)
- [PyTorchのSchedulerまとめ - 情弱大学生の独り言](http://katsura-jp.hatenablog.com/entry/2019/01/30/183501#StepLR)

