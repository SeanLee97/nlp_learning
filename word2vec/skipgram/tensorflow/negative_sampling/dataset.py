# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# dataset.py 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import re
import random
import numpy as np
import nltk
import jieba
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
		self.vocab = self.get_vocab(data)
		self.windows = []
		self.vocab.append(const.U_TOKEN)
		self.word2idx = {}
		self.idx2word = {}
		self.n_words = 0

		for word in self.vocab:
			if word not in self.word2idx:
				self.word2idx[word] = self.n_words
				self.idx2word[self.n_words] = word
				self.n_words += 1

		for sentence in data:
			# n-gram
			self.windows.extend(\
				list(\
					nltk.ngrams([const.D_TOKEN]*const.WIN_SIZE+sentence+[const.D_TOKEN]*const.WIN_SIZE, const.WIN_SIZE*2+1)\
				)\
			)

		dataset = []
		for window in self.windows:
			for i in range(const.WIN_SIZE*2+1):
				if i == const.WIN_SIZE or window[i] == const.D_TOKEN: 
					continue
				dataset.append((window[const.WIN_SIZE], window[i]))
		X_p, y_p = [], []
		for d in dataset:
			X_p.append(self.var_word(d[0]))
			y_p.append(self.var_word(d[1]))
		self.dataset = list(zip(X_p, y_p))

	def get_vocab(self, data):
		# [[]] -> []
		flatten = lambda l: [item.lower() for sublist in l for item in sublist]
		word_count = Counter(flatten(data))
		border = int(len(word_count)*0.01)
		stopwords = word_count.most_common()[:border]+list(reversed(word_count.most_common()))[:border]
		stopwords = [s[0] for s in stopwords]
		vocab = list(set(flatten(data))-set(stopwords))
		return vocab

	# @return batch data
	# @generator
	def batch_data(self, batch_size):
		random.shuffle(self.dataset)
		sidx = 0			# start index
		eidx = batch_size	# end index
		while eidx < len(self.dataset):
			batch = self.dataset[sidx:eidx]
			sidx = eidx
			eidx += batch_size
			yield batch

		if eidx >= len(self.dataset):
			batch = self.dataset[sidx: ]
			diff = eidx - len(self.dataset)
			inps, targets = zip(*batch)  # unzip
			inps = list(inps)
			targets = list(targets)
			diff_vec = [self.word2idx[const.U_TOKEN]]*diff
			inps = inps + diff_vec
			targets = targets + diff_vec
			inps = tuple(inps)
			targets = tuple(targets)
			batch = zip(inps, targets)
			yield batch

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
