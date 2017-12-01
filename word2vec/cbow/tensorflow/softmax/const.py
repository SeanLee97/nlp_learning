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

WIN_SIZE = 4		# window size
SKIP_WIN = 2

# nnwork
EMBEDDING_SIZE = 100
BATCH_SIZE = 128
EPOCH = 10000
LR_RATE = 0.0001

MODEL_PATH = './model/word2vec.bin'
