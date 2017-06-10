# iNatularist image loader


from PIL import Image
import os
import json
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def gen_list(prefix):
	ann_file = '%s2017.json'%prefix
	train_out = '%s.lst'%prefix
	# load annotations
	print('Loading annotations from: ' + os.path.basename(ann_file))
	with open(ann_file) as data_file:
		ann_data = json.load(data_file)

	# set up the filenames and annotations
	imgs = [aa['file_name'] for aa in ann_data['images']]
	im_ids = [aa['id'] for aa in ann_data['images']]
	if 'annotations' in ann_data.keys():
		# if we have class labels
		classes = [aa['category_id'] for aa in ann_data['annotations']]
	else:
		# otherwise dont have class info so set to 0
		classes = [0]*len(im_ids)

	idx_to_class = {cc['id']: cc['name'] for cc in ann_data['categories']}

	print('\t' + str(len(imgs)) + ' images')
	print('\t' + str(len(idx_to_class)) + ' classes')

	for index in range(10):
		path = imgs[index]
		target = str(classes[index])
		im_id = str(im_ids[index]-1)
		print(im_id + '\t' + target + '\t' + path)

	import pandas as pd
	from sklearn.utils import shuffle

	df = pd.DataFrame(classes)
	df[1] = imgs
	df = shuffle(df)

	df.to_csv(train_out, sep='\t', header=None, index=False)
	df = pd.read_csv(train_out, delimiter='\t', header=None)
	df.to_csv(train_out, sep='\t', header=None)

if __name__ == '__main__':
	set_names = ['train', 'val', 'test']
	for name in set_names:
		gen_list(name)
