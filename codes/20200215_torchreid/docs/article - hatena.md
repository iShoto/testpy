# Torchreid入門

Person ReIDライブラリーの[Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
がいい感じだったので簡単まとめておく。
[チュートリアル](https://kaiyangzhou.github.io/deep-person-reid/)
に色々な使い方が記載されているが、ここでは以下の3つについてまとめている。

1. 訓練  
[README.md](https://github.com/KaiyangZhou/deep-person-reid)の
`Get started: 30 seconds to Torchreid`。
Market1501で距離学習モデルを訓練する。
2. テスト  
[チュートリアル](https://kaiyangzhou.github.io/deep-person-reid/)の
`Test a trained model`。
1で訓練でしたモデルの精度を算出する。
3. ランキング結果の可視化  
[チュートリアル](https://kaiyangzhou.github.io/deep-person-reid/)の
`Visualize ranking results`。
1で訓練したモデルでqueryに対するgalleryの検索結果を可視化する。

コードは[こちら](https://github.com/iShoto/testpy/tree/master/codes/20200215_torchreid)。


## 1. 訓練

データ、モデル定義、最適化関数は、訓練だけでなくテストとランキング結果の可視化でも必要になるので、
これらを取得する共通メソッド`get_items()`を定義しておく。
また訓練ではschedulerも必要なので、それもリターンする。
これらを`train()`に渡してモデルを訓練する。

```python
# Training
datamanager, model, optimizer, scheduler = get_items()
train(datamanager, model, optimizer, scheduler)
```

データは`datamanager`、モデル定義は`model`、最適化関数は`optimizer`、
スケジューラーは`scheduler`に格納する。

- データ  
今回は訓練データとテストデータ共にmarket1501を利用する。
各々sourcesとtargetsに指定するが、他のデータ（との組合せ）を記載することも可能。

- モデル定義  
ベースモデルはResNet50を利用。
メトリックの指定方法など、具体的な設定方法はまだ勉強中。

- 最適化関数  
無難にAdamを利用。
`pytorch.optim`に実装されているのなら利用できる予感。

- スケジューラー  
シングルステップを指定。
`pytorch.optim.lr_scheduler`に実装されているのなら利用できる予感。


```bash
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
```

上で定義した4つ変数を引数として訓練を実施。
`engine`を作って`run`を実行する。
チェックポイントを`save_dir`に保存してくれる。
また訓練時は`test_only`を`False`にする。

```bash
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
```

訓練が開始されると次のように標準出力される。
GeForce GTX 1080だと、60エポックは1時間46分かかり、モデル精度はRank-1で85.2%となった。

```bash
> python .\main.py
Building train transforms ...
+ resize to 256x128
+ random flip
+ random crop (enlarge to 288x144 and crop 256x128)
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Building test transforms ...
+ resize to 256x128
+ to torch tensor of range [0, 1]
+ normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=> Loading train (source) dataset
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
=> Loading test (target) dataset
=> Loaded Market1501
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------


  **************** Summary ****************
  source            : ['market1501']
  # source datasets : 1
  # source ids      : 751
  # source images   : 12936
  # source cameras  : 6
  target            : ['market1501']
  *****************************************


=> Start training
Epoch: [1/1][10/404]    Time 0.188 (0.833)      Data 0.000 (0.505)      Loss 6.7096 (6.7458)    Acc 0.00 (0.62) Lr 0.000300     eta 0:05:28
Epoch: [1/1][20/404]    Time 0.190 (0.511)      Data 0.001 (0.253)      Loss 6.7111 (6.8223)    Acc 0.00 (0.94) Lr 0.000300     eta 0:03:16
Epoch: [1/1][30/404]    Time 0.187 (0.404)      Data 0.001 (0.169)      Loss 6.6187 (6.7905)    Acc 0.00 (1.15) Lr 0.000300     eta 0:02:31
...
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
Checkpoint saved to "../experiments/models/checkpoints/model.pth.tar-60"

Elapsed 1:45:48
```

## 2. テスト

テストとランキング結果の可視化のコードはほどんど同じ。
ロードした学習済みモデルの重みについて、精度評価するか距離計算するかの違い。
先に述べたようにデータ、モデル定義、最適化関数を引数として渡す。

```python
# Test
weight_path = '../experiments/models/model_market1501_resnet50.pth.tar-60'
torchreid.utils.load_pretrained_weights(model, weight_path)
test(datamanager, model, optimizer)
vis_rank(datamanager, model, optimizer)
```

`engine.run()`で`test_only`を`True`にするとテスト（精度評価）になる。

```python
def test(datamanager, model, optimizer):
	# Step 5: construct engine
	engine = torchreid.engine.ImageSoftmaxEngine(
		datamanager,
		model,
		optimizer=optimizer
	)

	# Step 6: run model training and test
	engine.run(
		test_only=True
```

訓練時、最後に保存したチェックポイントをロードしているので精度が同じになる。

```bash
Successfully loaded pretrained weights from "../experiments/models/model_market1501_resnet50.pth.tar-60"
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-2048 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-2048 matrix
Speed: 0.0317 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 68.4%
CMC curve
Rank-1  : 85.2%
Rank-5  : 93.6%
Rank-10 : 95.9%
Rank-20 : 97.3%
```


## 3. ランキング結果の可視化

ランキング結果の可視化でも2と同様に精度算出してくれる。
さらに画像検索の結果も保存してくれる。
`engine.run()`の`save_dir`に画像検索結果の保存先を指定して、
`visrank`を`True`にする。

```python
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
```

標準出力は次のようになる。

```bash
Successfully loaded pretrained weights from "../experiments/models/model_market1501_resnet50.pth.tar-60"
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-2048 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-2048 matrix
Speed: 0.0242 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 68.4%
CMC curve
Rank-1  : 85.2%
Rank-5  : 93.6%
Rank-10 : 95.9%
Rank-20 : 97.3%
# query: 3368
# gallery 15913
Visualizing top-10 ranks ...
- done 100/3368
- done 200/3368
- done 300/3368
- done 400/3368
- done 500/3368
...
```

検索結果は次の通り。
1列目のように得意なqueryは全正解だが、
2列目のように画像の一部がqueryになると著しく正解率が下がる。
また3列目のようにリュックと服の色の区別ができている訳でもないみたい。

![pic](../docs/images/0001_c1s1_001051_00.jpg)
![pic](../docs/images/0001_c2s1_000301_00.jpg)
![pic](../docs/images/0021_c1s1_002301_00.jpg)


## まとめ
様々なデータやモデルが気軽に非常に便利。
ただし、色々やれるが痒い所に手は届かないので、
これをベースに改良したい感じ。
