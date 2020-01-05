# MNISTで画像検索

前回「[MNISTで距離学習](http://testpy.hatenablog.com/entry/2020/01/04/225231)」
という記事を書いたが画像分類の域を出なかった。
距離学習と言えば画像検索なので、今回はそれをMNISTで行った。


## 概要
今回は[前回](http://testpy.hatenablog.com/entry/2020/01/04/225231)
訓練したMNISTの距離学習モデルを利用して画像検索を行う。
手順は次の通り。

1. データ準備
2. モデルロード
3. 特徴抽出
4. 距離算出

これらについて順番に説明する。
なお、コードは[ここ](https://github.com/iShoto/testpy/tree/master/codes/20200105_metric_learning_mnist_query_and_gallery)に置いてある。

## 1. データ準備
テスト時、画像分類ではテストデータを用意するが、
画像検索では検索画像となるQueryと検索対象画像群となるGalleryが必要になる。
そこでまずはテストデータをQueryとGalleryに分ける。
また後々のことを考えて、いったん画像に保存したものを読み込むようにする。

ここではメインファイルの`image_retrieval.py`で`make_query_and_gallery_from_mnist()`を呼び出す。
これにより、MNISTから1枚をQueryに100枚をGalleryにランダム選択して振り分ける。
各画像はQueryとGallery各々のディレクトリーに保存され、それらの情報はCSVファイルに記載される。

`dataset_dir`はMNISTを保存するディレクトリー、
`query_dir`はQuery画像を保存するディレクトリー、
`gallery_dir`はGallery画像を保存するディレクトリー、
`anno_path`はQueryとGalleryの画像情報を記載したCSVファイル
である。

```python
make_query_and_gallery_from_mnist(args.dataset_dir, args.query_dir, args.gallery_dir, args.anno_path)
```

`make_query_and_gallery_from_mnist()`では、
まず`make_query_and_gallery()`でMNISTの画像をQueryとGalleryに振り分けて、
次に`make_anno_file()`でQueryとGalleryの画像情報をCSVに保存する。
これらはMNISTデータ処理専用の`mnist_data.py`で行う。

```python
def make_query_and_gallery_from_mnist(dataset_dir, query_dir, gallery_dir, anno_path):
	mnist_data.make_query_and_gallery(dataset_dir, query_dir, gallery_dir)
	mnist_data.make_anno_file(query_dir, gallery_dir, anno_path)
```

以下から`mnist_data.py`。
`make_query_and_gallery()`は次の通り。
transformしたMNISTを取得して、Query画像1枚とGallery画像100枚をランダムに選択後、
各々のディレクトリーに画像として保存している。
保存前の画像はtransfromで正規化した後なので0～255になっていないが、
`scipy.misc.imsave()`を使うと0～255にして保存してくれる。

```python
def make_query_and_gallery(dataset_dir, query_dir, gallery_dir):
	# 
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	testset = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
	q_idx = random.choice(range(len(testset)))
	g_idxs= random.sample(range(len(testset)), 100)
	
	# Save query image.
	if os.path.exists(query_dir) == True:
		shutil.rmtree(query_dir)
	os.makedirs(query_dir)
	q_img, q_label = testset[q_idx]
	scipy.misc.imsave(query_dir+'{}_{}.png'.format(q_label, q_idx), q_img.numpy()[0])
	
	# Save gallery images.
	if os.path.exists(gallery_dir) == True:
		shutil.rmtree(gallery_dir)
	os.makedirs(gallery_dir)
	for g_idx in g_idxs:
		g_img, g_label = testset[g_idx]
		scipy.misc.imsave(gallery_dir+'{}_{}.png'.format(g_label, g_idx), g_img.numpy()[0])
```

`make_anno_file()`では、Query/Gallery、
画像名、画像パス、ラベル名、IDを記載したCSVファイルを作成して保存する。

```python
def make_anno_file(query_dir, gallery_dir, anno_path):
	annos = []
	annos += __set_annos(query_dir, 'query')
	annos += __set_annos(gallery_dir, 'gallery')
	df = pd.DataFrame(annos)
	df.to_csv(anno_path, index=False)


def __set_annos(img_dir, data_type):
	annos = []
	for d in os.listdir(img_dir):
		dic = {}
		dic['data_type'] = data_type
		dic['img_name'] = d
		dic['img_path'] = img_dir + d
		dic['label'] = d.split('_')[0]
		dic['id'] = d.split('.')[0].split('_')[1]
		annos.append(dic)

	return annos
```

最後に画像検索用のDataLoaderを作成する。
先ほど作成したQueryとGalleryのCSVファイルを用いて、
各々の画像ローダーが作成できる仕様にしている。
また、画像とラベルの他に画像パスも返すようにしている。

```python
class ReIDDataset(Dataset):
	def __init__(self, anno_path, data_type, transform=None):
		df_all = pd.read_csv(anno_path)
		self.df = df_all[df_all['data_type']==data_type].reset_index(drop=True)  # Filter data by query or gallery.
		self.transform = transform


	def __len__(self):
		return len(self.df)


	def __getitem__(self, idx):
		img_path = self.df.loc[idx, 'img_path']
		assert os.path.exists(img_path)
		image = io.imread(img_path)
		label = self.df.loc[idx, 'label']
		img_path = self.df.loc[idx, 'img_path']
		if self.transform:
			image = self.transform(image)
		
		return image, label, img_path
```

画像分類の時にtrain_loader、test_loader、classesを返していたのと同様、
画像検索の時はquery_loader、gallery_loader、classesを返す。


```python
def load_query_and_gallery(anno_path, img_show=False):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	# Query
	query_dataset = ReIDDataset(anno_path, 'query', transform)
	query_loader = DataLoader(query_dataset, batch_size=len(query_dataset), shuffle=False)
		
	# Gallery
	gallery_dataset = ReIDDataset(anno_path, 'gallery', transform)
	#gallery_loader = DataLoader(gallery_dataset, batch_size=len(gallery_dataset), shuffle=True)
	gallery_loader = DataLoader(gallery_dataset, batch_size=8, shuffle=True)
	
	# Class
	classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	# debug
	print('num query: {}, num gallery: {}'.format(len(query_dataset), len(gallery_dataset)))
	print('')
	if img_show == True:
		show_data(gallery_loader)

	return query_loader, gallery_loader, classes
```


## 2. モデルロード

再びメインファイルの`image_retrieval.py`。
モデルのロードは、`model.load_state_dict(torch.load(args.model_path))`で行う。
学習済みモデルは、[前回の記事](http://testpy.hatenablog.com/entry/2020/01/04/225231)を参考に生成する。
今回はテストなので`eval()`で評価モードにしておく。

```python
# Set device, GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Net().to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()
```

## 3. 特徴抽出

`1. データ準備`で説明した`query_loader`を利用する。
と言っても画像は1枚だけなので、学習済みモデルに通して特徴を取得するだけ。
`model`の出力は2次元特徴と予測結果なるが、今回は前者のみを利用する。

```python
# Query
for i, (query_img, query_label, query_path) in enumerate(query_loader):
	with torch.no_grad():
		query_img = query_img.to(device)
		query_feat, pred = model(query_img)
```

Galleryは100枚画像があるが、
`gallery_loader`のバッチサイズは8にしているので、
各情報をリストに格納して、最後にconcatinateする。

```python
# Gallery
gallery_feats = []
gallery_labels = []
gallery_paths = []
for i, (g_imgs, g_labels, g_paths) in enumerate(gallery_loader):
	with torch.no_grad():
		g_imgs = g_imgs.to(device)
		g_feats_temp, preds_temp = model(g_imgs)
		gallery_feats.append(g_feats_temp)
		gallery_labels.append(g_labels)
		gallery_paths += list(g_paths)  # Data type of g_paths is tuple.
gallery_feats = torch.cat(gallery_feats, 0)
gallery_labels = torch.cat(gallery_labels, 0)
```

## 4. 距離算出

`3. 特徴抽出`でQueryの特徴`query_feat`とGalleryの特徴`gallery_feats`が取得できた。
`query_feat`と`gallery_feats`の各特徴との距離を算出するためにコサイン類似度を利用する。

```python
dist_matrix = cosine_similarity(query_feat, gallery_feats)
```

コサイン類似の実装は次の通り。

```python
def cosine_similarity(qf, gf):
	epsilon = 0.00001
	dist_mat = qf.mm(gf.t())
	qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True) #mx1
	gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True) #nx1
	qg_normdot = qf_norm.mm(gf_norm.t())

	dist_mat = dist_mat.mul(1/qg_normdot).cpu().numpy()
	dist_mat = np.clip(dist_mat, -1+epsilon,1-epsilon)
	dist_mat = np.arccos(dist_mat)

	return dist_mat
```

QueryとGalleryの各距離、Galleryのラベル名と画像パスをセットにした
DataFrameを作成して、距離でソートする。
これにより、距離の近い順にラベルが表示される。

```python
# Organize ReID ranking.
lis = []
for i in range(len(gallery_paths)):
	dic = {}
	dic['dist'] = dist_matrix.tolist()[0][i]
	dic['label'] = np.array(gallery_labels).tolist()[i]
	dic['img_path'] = gallery_paths[i]
	lis.append(dic)
df = pd.DataFrame(lis)
df = df.sort_values(by=['dist'], ascending=True)
df = df.reset_index(drop=True)
```

以下は実行結果。
Queryが9で、Galleryも9がindexの0から8まで占めている。

```sh
$ python image_retrieval.py
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

labelをカウントすると、ラベルが9の画像は9枚あることが分かる。
よって、Galleryにあるラベル9の全画像を検索上位に持ってくることができたことが分かる。

```sh
1    15
7    13
0    12
8    11
4    10
3    10
9     9  # <- this
6     7
2     7
5     6
```


## 参考文献
- [A Tiny Person ReID Baseline - github](https://github.com/lulujianjie/person-reid-tiny-baseline)
- [Shows image with specific index from MNIST dataset - pytorch](https://discuss.pytorch.org/t/shows-image-with-specific-index-from-mnist-dataset/29406)
- [Saving a Numpy array as an image - stackoverflow](https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image)
- [機械学習のお勉強（自作データセットでCNN by pytorch） - 空飛ぶロボットのつくりかた](http://robonchu.hatenablog.com/entry/2017/10/23/173317)
- [pytorchによる画像分類入門 - Qiita](https://qiita.com/sheep96/items/0c2c8216d566f58882aa)
- [PyTorch transforms/Dataset/DataLoaderの基本動作を確認する - Qiita](https://qiita.com/takurooo/items/e4c91c5d78059f92e76d)
