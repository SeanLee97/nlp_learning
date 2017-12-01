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

import torch
from torch.autograd import Variable

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

		# data sample
		data_split = len(data) // 10
		neg_data = data[-data_split:]
		data = data[:1-data_split]
		flatten = lambda l: [item.lower() for sublist in l for item in sublist]

		self.neg_vocab = list(set(flatten(neg_data)))

		word_count = Counter(flatten(data))
		self.word2idx = {const.U_TOKEN: 0}
		self.n_words = 1
		for word, _ in word_count.items():
			self.word2idx[word] = self.n_words
			self.n_words += 1
		self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys())) 
		self.vocab = list(self.word2idx.keys())

		# unigram_table
		vocab_total_words = sum([c for w, c in word_count.items() if w not in self.neg_vocab])
		self.unigram_table = []
		for v in self.vocab:
			self.unigram_table.extend([v]*int(((word_count[v]/vocab_total_words)**(3/4))/const.Z))

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

	def negative_sampling(self, targets):
		batch_size = targets.size(0)
		neg_samples = []
		for i in range(batch_size):
			sample = []
			target_idx = targets[i].data.tolist()[0]
			while len(sample) < const.NEG:
				if self.word2idx == target_idx:
					continue
				sample.append(random.choice(self.unigram_table))
			neg_samples.append(Variable(torch.LongTensor(self.var_sentence(sample))).view(1, -1))
		return torch.cat(neg_samples)

	# @input sentence [w1, w2, ... , wn]
	def var_sentence(self, sentence):
		idxs = list(map(lambda w: self.word2idx[w] if w in self.vocab else self.word2idx[const.U_TOKEN], sentence))
		return idxs

	# @input word
	def var_word(self, word):
		idx = [self.word2idx[const.U_TOKEN]]
		if word in self.word2idx:
			idx = [self.word2idx[word]]
		return idx
