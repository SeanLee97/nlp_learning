# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: 语言模型
 UniGram 
 BiGram
 TriGram
 GramUtil - 工具函数
@author: Sean QQ: 929325776
'''

import math
import const
from processing import *

'''
@function calc_prob 		计算条件概率，这里使用最大似然估计(max-likelihood estimate)去计算概率
@function calc_sentence_prob	计算句子的条件概率
'''
class UnGram(object):
	def __init__(self, sentences, smooth = None):
		self.undict, self.total = build_undict(sentences)
		self.smooth = smooth

	def calc_prob(self, word):
		prob = 0
		if self.smooth != None:
			prob = self.smooth(word, undict=self.undict, total=self.total)
		else:
			if word in self.undict:
				prob = float(self.undict[word]) / self.total
		return prob

	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		for word in sentence:
			if word != const.START_TOKEN and word != const.END_TOKEN:
				word_prob = self.calc_prob(word)
				if word_prob != 0:
					prob_log_sum += math.log(word_prob, 2)
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum

	def sort_vocab(self):
		vocabs = list(self.undict.keys())
		vocabs.remove(const.START_TOKEN)
		vocabs.remove(const.END_TOKEN)
		vocabs.sort()
		vocabs.append(const.UNK)
		vocabs.append(const.START_TOKEN)
		vocabs.append(const.END_TOKEN)
		return vocabs

class BiGram(UnGram):
	def __init__(self, sentences, smooth = None):
		UnGram.__init__(self, sentences, smooth)
		self.bidict = build_bidict(sentences)

	def calc_prob(self, *args):
		if len(args) != 2:
			raise ValueError('two words is required')

		prob = 0
		if self.smooth != None:
			prob = self.smooth(args[0], args[1], bidict=self.bidict, undict=self.undict)
		else:
			if args in self.bidict and args[0] in self.undict:
				return float(self.bidict[args]) / self.undict[args[0]]
		return prob

	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		prev_word = None
		for word in sentence:
			if prev_word != None:
				word_prob = self.calc_prob(prev_word, word)
				prob_log_sum += word_prob
			prev_word = word
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum


class TriGram(BiGram):
	def __init__(self, sentences, smooth = None):
		BiGram.__init__(self, sentences, smooth)
		self.tridict = build_tridict(sentences)

	def calc_prob(self, *args):
		if len(args) != 3:
			raise ValueError('three words is required')

		prob = 0
		if self.smooth != None:
			prob = self.smooth(args[0], args[1], args[2], tridict=self.tridict, bidict=self.bidict, undict=self.undict)
		else:
			bitup = (args[0], args[1])				
			if args in self.tridict and bitup in self.bidict:
				return float(self.tridict[args]) / self.bidict[bitup]
		return prob

	def calc_sentence_prob(self, sentence, prob_log=True):
		prob_log_sum = 0
		prev_stack = []
		for word in sentence:
			if len(prev_stack) < 2:
				prev_stack.append(word)
			elif len(prev_stack) == 2:
				word_prob = self.calc_prob(prev_stack[0], prev_stack[1], word)
				prob_log_sum += word_prob
				prev_stack[0] = prev_stack[1]
				prev_stack[1] = word
		return math.pow(2, prob_log_sum) if prob_log else prob_log_sum

'''
@function: calc_xxgram_count   主要用来统计语料库中词的总数
@function: print_xxgram_probas 格式化输出概率 
'''
class GramUtil(object):

	@staticmethod
	def calc_ungram_count(sentences):
		count = 0
		for sentence in sentences:
			# except START_TOKEN and END_TOKEN
			count += len(sentence) - 2
		return count

	@staticmethod
	def calc_bigram_count(sentences):
		count = 0
		for sentence in sentences:
			count += len(sentence) - 1
		return count

	@staticmethod
	def calc_trigram_count(sentences):
		count = 0
		for sentence in sentences:
			count += len(sentence)
		return count

	@staticmethod
	def print_ungram_probs(model, vocabs):
		for vocab in vocabs:
			if vocab != const.START_TOKEN and vocab != const.END_TOKEN:
				print("{} \t {}".format(vocab if vocab != const.UNK else 'UNK', model.calc_prob(vocab)))

	@staticmethod
	def print_bigram_probs(model, vocabs):
		print("\t\t", end="")
		for vocab in vocabs:
			if vocab != const.START_TOKEN:
				print(vocab if vocab != const.UNK else "UNK", end="\t\t")
		print("")
		for vocab in vocabs:
			if vocab != const.END_TOKEN:
				print(vocab if vocab != const.UNK else "UNK", end="\t\t")
				for vocab2 in vocabs:
					if vocab2 != const.START_TOKEN:
						print("{0:.3f}".format(model.calc_prob(vocab, vocab2)), end="\t\t")
				print("")

	@staticmethod
	def print_trigram_probs(model, vocabs):
		print("\t\t", end="")
		for vocab in vocabs:
			if vocab != const.START_TOKEN:
				print(vocab if vocab != const.UNK else "UNK", end="\t")
		print("")
		for vocab in vocabs:
			if vocab != const.END_TOKEN:
				for vocab2 in vocabs:
					if vocab2 != const.START_TOKEN and vocab != const.UNK and vocab2 != const.UNK and vocab2 != const.END_TOKEN:
						print(vocab, vocab2 if vocab2 != const.UNK else "UNK", end="\t\t")
						for vocab3 in vocabs:
							if vocab3 != const.END_TOKEN:
								print("{0:.3f}".format(model.calc_prob(vocab, vocab2, vocab3)), end="\t")
						print("")
