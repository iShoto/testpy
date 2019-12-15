


## 初級版 Deep Metric Learning を PyTorch 初心者が写経

最近、Deep なコードを自分で実装しておらず、 PFN が PyTorch に移行した事もあり、これから日本全体が PyTorch に移行するのを見据えて、自分も忘れかけてる Keras から PyTorch へ移行しようと思い、 ArcFace で流行ってきた感がある Deep Metric Learning を題材に写経しました。

### 1. モデル

#### nn.PReLU()

```py
import torch.nn as nn
self.prelu1_1 = nn.PReLU()
```

ReLUの改良の改良。
LeakyReLU で `x < 0` の時に `y < 0` することで学習が進みやすくなったものの、
パラメーター α が増えたため、それを減らすべく学習させることにしたのがPReLU。
ちなみにPReLUは、"a Parametric Rectified Linear Unit" の略。

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

- [LeakyRelu活性化関数 - Thoth Children](http://www.thothchildren.com/chapter/59b93f7575704408bd4300f2)
- [PRelu活性化関数 - Thoth Children](http://www.thothchildren.com/chapter/59b940a475704408bd4300f8)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification - arXiv](https://arxiv.org/abs/1502.01852)


### view

```
x = x.view(-1, 128*3*3)
```

np.reshapeと同じ。
第一引数が-1のとき、第二引数に自動調整してくれる。
上記の場合だと、x の shape が (3, 3, 128) だと思うので、
1次元に変換している。
しかし、画像の入力サイズによって、3x3にならないのでは。。。
そう言えば、入力サイズってどう定義してるんだろ、PyTroch。

### 2. Loss

#### nn.NLLLoss()
Negative Log-Likelihood (NLL) Lossの略。
softmaxの最大値は結果の確信度を表すが、それをマイナスの対数で取った値となる。
NLL Lossにより、高い確信度であれば低いロス、低い確信度であれば高いロスを割り当てることができる。

- [Understanding softmax and the negative log-likelihood - Lj Miranda](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)


#### @staticmethod
静的関数


- [Difference between 'ctx' and 'self' in python? - stack overflow](https://stackoverflow.com/questions/49516188/difference-between-ctx-and-self-in-python)




### 3. Training

#### optimizer & loss

訓練中、optimierとlossはバッチごとに次の手順を踏む。

```
optimizer.zero_grad()  # 勾配の初期化
loss.backward()  # 勾配の計算
optimizer.step()  # パラメータの更新
```

- [実践Pytorch - Qiita](https://qiita.com/perrying/items/857df46bb6cdc3047bd8)


### 5. Visualize

#### plt.ion()
インタラクティブモードをオンにする。
`plt.draw()`で描画するときに必要。
`plt.savefig()`で描画結果を保存するなら不要。

- [plt.ion() すると描画されないことがある - Qiita](https://qiita.com/halm/items/becdc1e1a456562f21c8)


#### torchvision.utils.make_grid()

train_loader のバッチを簡単に可視化できる。

```py
images, labels = iter(train_loader).next()  # train_loader のミニバッチの image を取得
img = torchvision.utils.make_grid(images, nrow=12, padding=1)  # nrom*nrom のタイル形状の画像を作る
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 画像を matplotlib 用に変換
plt.show()
```

- [PyTorch まずMLPを使ってみる - cedro-blog](http://cedro3.com/ai/pytorch-mlp/)