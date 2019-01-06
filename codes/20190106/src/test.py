# coding: utf-8
from PIL import Image
import sys
import pyocr
import pyocr.builders
import os
from tqdm import trange

# Set Netflix subtitle image directory.
img_dir = '../data/planetes/PLANETES.S01E02.WEBRip.Netflix/'
#img_dir = '../data/'

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
