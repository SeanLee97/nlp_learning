# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: 加载语料，并加入起始标记<s></s>
@author: Sean QQ: 929325776
'''
import re
from processing import build_sentences

def load_dataset(file_path):
	with open(file_path, "r") as f:
		return build_sentences([re.split("\s+", line.rstrip('\n')) for line in f])
