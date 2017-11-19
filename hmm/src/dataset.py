# !/usr/bin/env python3
# -*- coding: utf-8 -*-

def load_data(file_path):
	datas, tags = [], []
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			splits = line.split(' ')
			data, tag = [], [] 
			for part in splits:
				parts = part.split('/')
				data.append(parts[0])
				tag.append(parts[1])
			datas.append(data)
			tags.append(tag)
	return datas, tags

