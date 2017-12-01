# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# locate: Shanxi university, Taiyuan, China  #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

S_TOKEN = '<s>'		# start token
E_TOKEN = '</s>'	# end token
U_TOKEN = '<u>'		# unknown token
D_TOKEN = '<d>'		# dummy token

WIN_SIZE = 3		# window size

# nnwork
EMBEDDING_SIZE = 30
BATCH_SIZE = 128
EPOCH = 1000
LR_RATE = 0.001
NEG = 10 # Num of Negative Sampling
