# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: 模型评估
@author: Sean QQ: 929325776
'''

import math

# 计算困惑度
def perplexity(model, sentences, cal_gram_func):
	# gram_count 词的总数，对应教程中的 M
	gram_count = cal_gram_func(sentences)
	prob_log_sum = 0
	for sentence in sentences:
		try:
			prob_log_sum -= math.log(model.calc_sentence_prob(sentence), 2)
		except:
			prob_log_sum -= float('-inf')
		return math.pow(2, prob_log_sum/gram_count)

