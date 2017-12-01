# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# cbow   	 			    				 #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cbow(nn.Module):
	def __init__(self, input_size, projection_size):
		super(Cbow, self).__init__()
		self.V = nn.Embedding(input_size, projection_size)
		self.U = nn.Embedding(input_size, projection_size)
		self.logsigmoid = nn.LogSigmoid()
		
		initrange = (2.0 / (input_size + projection_size))**5
		self.V.weight.data.uniform_(-initrange, initrange)
		self.U.weight.data.uniform_(-0.0, 0.0)  # zero

	def forward(self, center_words, target_words, neg_words):
		v = self.V(center_words)  # batch_size x 1 x projection_size
		u = self.U(target_words)	# batch_size x 1 x projection_size
		u_neg = -self.U(neg_words)

		pos_score = u.bmm(v.transpose(1, 2)).squeeze(2)    # batch_size x 1
		neg_score = torch.sum(u_neg.bmm(v.transpose(1, 2)).squeeze(2), 1).view(neg_words.size(0), -1)	# batch_size x input_size
		
		return self.loss(pos_score, neg_score)
		
	def loss(self, pos_score, neg_score):
		loss = self.logsigmoid(pos_score) + self.logsigmoid(neg_score)
		return -torch.mean(loss)

	def pred(self, inp):
		return self.V(inp)

