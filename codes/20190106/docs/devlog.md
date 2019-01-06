# 2019-01-07 PythonでOCR

[Netflixの字幕がダウンロードできる](https://ameblo.jp/macgyverisms/entry-12378152858.html)と知ったので実行してみたが、
日本語字幕が画像だったため、PythonでOCRにかけて文字列に変換した。結果は微妙。

## Install
1. pyorc-0.5.3
2. tesseract-ocr-setup-3.05.02-20180621

2は、[Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)から、
`tesseract-ocr-setup-3.05.02-20180621.exe`をダウンロード。
最初3.02を使っていたが、精度が低かったため、3.05を選択。
4.00が出てるので、そちらの方がいいかも。
exe実行時のオプションで、ダウンロードする言語を選択するとき、日本語を必ず選択すること。
なお、OSはWindows10。

## Code
以下のコードを実行すると、字幕画像を文字列にしてファイルに出力してくれる。
OCRで解析したい画像字幕のフォルダはimg_dirに指定する。
文字列にした字幕ファイルもここに保存される。
なお、コードは[github]()にも上げてます。

```python
# coding: utf-8
from PIL import Image
import sys
import pyocr
import pyocr.builders
import os
from tqdm import trange

# Set Netflix subtitle image directory.
img_dir = '../data/planetes/PLANETES.S01E02.WEBRip.Netflix/'

# Get a tool.
tools = pyocr.get_available_tools()
if len(tools) == 0:
	print("No OCR tool found")
	sys.exit(1)
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))

# Image to string.
txts = []
img_names = [f for f in os.listdir(img_dir) if f.split('.')[-1].lower() in ('png')]
img_names = sorted(img_names)
print(img_names)
for i in trange(len(img_names), desc='img 2 str'):
	txt = tool.image_to_string(
		Image.open(img_dir+img_names[i]),
		lang='jpn',
		builder=pyocr.builders.TextBuilder()
	)
	print(txt)
	txts.append(txt)

# Save the subtitle.
subs = open(img_dir+'subs.txt', 'w')
subs.write('\n'.join(txts))
subs.close()
```

## Result

複雑な漢字が苦手らしい。
また、他の画像でも試したが、精度が画像の大きさとも相関している模様。

- 画像  
![pic](ttpi000003.png)

- 文字列  
"""
ステ一 ンヨノ建L ロ時に出た廃某 武物
"""

- 画像  
![pic](ttpi000075.png)

- 文字列  
"""
これなんか
かなり当たるようですよ
"""

## References
- [Netflixの字幕をダウンロードする方法](https://ameblo.jp/macgyverisms/entry-12378152858.html)
- [pipを使わずwindows10にtesseractとPyOCRをインストールする方法](https://haitenaipants.hatenablog.com/entry/2018/06/02/193554)
- [PyOCR】画像から日本語の文字データを抽出する](https://qiita.com/mczkzk/items/393abc70836b9bde2f60)
- [Windows で Tesseract 3.0.5 を使ってみる](https://www.kunihikokaneko.com/dblab/licenseplate/tesseract.html)
