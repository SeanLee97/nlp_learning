# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# dataset.py 			             #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import re
import random
import nltk
import jieba
import torch
from torch.autograd import Variable
from collections import defaultdict, Counter


if torch.cuda.is_available():
	FloatTensor = torch.cuda.FloatTensor
	LongTensor = torch.cuda.LongTensor
	ByteTensor = torch.cuda.ByteTensor

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
		self.vocab, self.neg_vocab, self.unigram_table = self.get_vocab(data)
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
				if window[i] in self.neg_vocab or window[const.WIN_SIZE] in self.neg_vocab:
					continue
				if i == const.WIN_SIZE or window[i] == const.D_TOKEN: 
					continue
				dataset.append((window[const.WIN_SIZE], window[i]))
		X_p, y_p = [], []
		for d in dataset:
			X_p.append(self.var_word(d[0]).view(1,-1))
			y_p.append(self.var_word(d[1]).view(1,-1))
		self.dataset = list(zip(X_p, y_p))

	def get_vocab(self, data, min_count=3, Z=0.01):
		# [[]] -> []
		flatten = lambda l: [item.lower() for sublist in l for item in sublist]
		word_count = Counter(flatten(data))
		neg_vocab = [w for w, c in word_count.items() if c < min_count]
		vocab = list(set(flatten(data))-set(neg_vocab))
		vocab_total_words = sum([c for w, c in word_count.items() if w not in neg_vocab])
		unigram_table = []
		for v in vocab:
			unigram_table.extend([v]*int(((word_count[v]/vocab_total_words)**(3/4))/Z))
		return vocab, neg_vocab, unigram_table

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
			neg_samples.append(self.var_sentence(sample).view(1, -1))
		return torch.cat(neg_samples)

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
			yield batch

	# @input sentence [w1, w2, ... , wn]
	# @return Variable
	def var_sentence(self, sentence):
		idxs = list(map(lambda w: self.word2idx[w] if w in self.word2idx.keys() \
				else self.word2idx[const.U_TOKEN], sentence))
		return Variable(torch.LongTensor(idxs))

	# @input word
	# @return Variable
	def var_word(self, word):
		return Variable(torch.LongTensor([self.word2idx[word]]) if word in self.word2idx.keys() \
				else torch.LongTensor([self.word2idx[const.U_TOKEN]]))
