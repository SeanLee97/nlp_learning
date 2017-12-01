# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

S_TOKEN = '<s>'		# start token
E_TOKEN = '</s>'	# end token
U_TOKEN = '<u>'		# unknown token
D_TOKEN = '<d>'		# dummy token

WIN_SIZE = 4		# window size
SKIP_WIN = 2		# skip window siaze
Z = 0.01

# nnwork
EMBEDDING_SIZE = 100
BATCH_SIZE = 128
EPOCH = 10000
LR_RATE = 0.001
NEG = 10 # Num of Negative Sampling
