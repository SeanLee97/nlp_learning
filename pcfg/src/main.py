# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# main.py    	                             #
# author: sean lee                           #
# qq: 929325776							     #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

from pcfg import PCFG 

parser = PCFG()
parser.fit('./corpus/toy/train.txt')
parser.parse("the man saw the dog")
'''
print(parser.N_dict)
print(parser.NR_dict)
print(parser.TR_dict)
'''