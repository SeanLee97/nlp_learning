# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import torch

# tokens
unk = '<unk>'
pad = '<pad>'
sos = '<s>'
eos = '</s>'

#nnwork
lr_rate = 0.001
batch_size = 16
hidden_size = 128
epochs = 10
task_id = 5	# 与bAbI/en-10k 中的task匹配

use_cuda = torch.cuda.is_available()
