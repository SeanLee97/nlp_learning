# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# skipgram	 			                     #
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

		self.V.weight.data.uniform_(-1.0, 1.0)
		self.U.weight.data.uniform_(0.0, 0.0)  # zero

	def forward(self, center_words, target_words, out_words):
		v = self.V(center_words)  # batch_size x win_size x projection_size
		u = self.U(target_words)	# batch_size x 1 x projection_size
		u_actual = self.U(out_words) # batch_size x input_size x projection_size

		scores = u.bmm(v.transpose(1, 2)).squeeze(2)    # batch_size x win_size
		norm_scores = u_actual.bmm(v.transpose(1, 2)).squeeze(2)	# batch_size x input_size
		return self.nll_loss(scores, norm_scores)
		
	def nll_loss(self, scores, norm_scores):
		#
		softmax = torch.exp(scores)/torch.sum(torch.exp(norm_scores),1).unsqueeze(1)
		return -torch.mean(torch.log(softmax))

	def pred(self, inp):
		return self.V(inp)

