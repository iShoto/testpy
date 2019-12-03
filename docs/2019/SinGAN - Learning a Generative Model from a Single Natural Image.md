## 0. メタデータ
- Title
	- SinGAN: Learning a Generative Model from a Single Natural Image
- Link
	- https://arxiv.org/abs/1905.01164
- Authers
	- Tamar Rott Shaham, Tali Dekel, Tomer Michaeli
- Conference
	- ICCV 2019

## 1. 概要
- ICCV 2019 Best Paper
- 1枚の画像に対して、様々な処理を実現する手法
- 処理内容は、Super Reslution、Paint to Image、Harmonization、Single Image Annimation、Editing
- 生成画像のスケールを拡大するGANのピラミッドから構成されるモデル
- 入力画像と入力スケールを調整することで目的の処理を実現する

## 2. 先行研究との差異
- Deep Painterly Harmonizationでは、写真の特徴を捉えるのが難しかったが、SinGANではそれを克服している

## 3. 手法

- $G_n$ で $\tilde{x}_n$ を生成して、$D_n$ で $x_n$ と区別する
- $x_n$ は $x$ から複数バッチを作成するが、効果的なサイズは右端時の黄色い矩形

- 各スケールの $G_n$ は、$D_n$ が $x_n$ と見分けられないような $\tilde{x}_n$ を生成するよう学習する
- スケールによって見分けるバッチサイズが異なり、スケールアップするに従い、バッチサイズは小さくなる

- 1つの画像から複数のパッチを取得することで、データセットのように扱い、統計をとる

- 画像セットではなく、1画像のパッチセットを学習に利用
- それ以外は、cGANと概念的には同じ
- 画像の複数の異なるスケールから、複雑な画像構造の統計値を取得する

## 4. 評価
- パッチでモデル訓練しているので、反射や影の出力が自然に実現できる
- 比較的細かい層から開始すると大枠のコンテキストが維持される

- Single Image Frechet Inception Distance(SIFID)の説明
- 1枚ずつのペア画像でもFIDの算出が可能
- SIFIDが低いほど、リアル画像に近い

## 5. 所感
- SIFID
- Deep Paintlery Harmonizationでできなかったリアル画像の特徴抽出

