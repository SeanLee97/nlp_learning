# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# main.py 									 #
# author: sean lee						 	 #
# email: lxm_0828@163.com					 #
#--------------------------------------------#

import argparse
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('-train', action='store_true', default=False, help='train model')
parser.add_argument('-test', action='store_true', default=False, help='test model')
args = parser.parse_args()

from dataset import Corpus, load_data
from cbow import Cbow

if __name__ == '__main__':
	
	data = list(load_data())
	corpus = Corpus(data)
	cbow = Cbow(corpus)


	if args.train:
		cbow.train()
	elif args.test:
		word = input('Input word> ')
		print(cbow.test(word))