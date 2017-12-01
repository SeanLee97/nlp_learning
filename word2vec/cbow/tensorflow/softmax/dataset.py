# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# dataset.py 		                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import re
import random
import numpy as np
import nltk
import jieba
import collections
from collections import defaultdict, Counter

def rm_sign(string):
	string = re.sub("[\.\!_,\$\(\)\"\'\]\[！!\?，。？、~@#￥……&]+", "", string) 
	return string

def load_data(corpus_dir = '../../../corpus/articles.txt'):
	with open(corpus_dir, 'r') as f:
		for line in f:
			line = line.strip()
			if len(line) == 0:
				continue
			yield jieba.lcut(rm_sign(line))

class Corpus(object):
	def __init__(self, data):
		flatten = lambda l: [item.lower() for sublist in l for item in sublist]
		word_count = Counter(flatten(data)).most_common()
		self.word2idx = {const.U_TOKEN: 0}
		self.n_words = 1
		for word, _ in word_count:
			self.word2idx[word] = self.n_words
			self.n_words += 1
		self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys())) 
		self.vocab = list(self.word2idx.keys())

	# @return batch data
	# @generator
	def batch_data(self):
		batch_size = const.BATCH_SIZE * const.WIN_SIZE
		data = self.vocab
		data_index = 0
		assert batch_size % const.WIN_SIZE == 0
		assert const.WIN_SIZE <= 2 * const.SKIP_WIN
		batch = np.ndarray(shape=(batch_size), dtype=np.int32)
		labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
		span = 2 * const.SKIP_WIN + 1 # [ const.SKIP_WIN target const.SKIP_WIN ]
		buffers = collections.deque(maxlen=span)
		for _ in range(span):
			buffers.append(data[data_index])
			data_index = (data_index + 1) % len(data)
		for i in range(batch_size // const.WIN_SIZE):

			target = const.SKIP_WIN  # target label at the center of the buffers
			targets_to_avoid = [const.SKIP_WIN]
			for j in range(const.WIN_SIZE):
				while target in targets_to_avoid:
					target = random.randint(0, span - 1)
				targets_to_avoid.append(target)
				batch[i * const.WIN_SIZE + j] = self.var_word(buffers[const.SKIP_WIN])[0]
				labels[i * const.WIN_SIZE + j, 0] = self.var_word(buffers[target])[0]
			buffers.append(data[data_index])
			data_index = (data_index + 1) % len(data)
		label_CBOW = []
		context_CBOW = []
		for i in range(0,len(batch), const.WIN_SIZE):
			label_CBOW.append(batch[i])
			context_CBOW.append([l[0] for l in labels[i:i+const.WIN_SIZE]])
		return np.array(context_CBOW), np.array(label_CBOW).reshape(batch_size // const.WIN_SIZE, 1)

	# @input sentence [w1, w2, ... , wn]
	def var_sentence(self, sentence):
		idxs = list(map(lambda w: self.word2idx[w] if w in self.word2idx.keys() \
				else self.word2idx[const.U_TOKEN], sentence))
		return idxs

	# @input word
	def var_word(self, word):
		idx = [self.word2idx[const.U_TOKEN]]
		if word in self.word2idx:
			idx = [self.word2idx[word]]
		return idx
