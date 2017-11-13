# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@description: 平滑估计计算
@author: Sean QQ: 929325776
'''

class Smooth(object):
	@staticmethod
	def discounting(*args, **kwargs):
		discount_value = 0.5
		if 'discount_value' in kwargs:
			discount_value = kwargs['discount_value']
		if len(args) == 1:
			if 'undict' not in kwargs:
				raise ValueError('undict is required')
			if 'total' not in kwargs:
				raise ValueError('total (words count in sentences) is required')
			undict = kwargs['undict']
			total = kwargs['total']
			word = args[0]
			if word in undict:
				return float(undict[word] - discount_value) / total
		if len(args) == 2:
			if 'bidict' not in kwargs and 'undict' not in kwargs:
				raise ValueError('bidict and undict is required')
			bidict = kwargs['bidict']
			undict = kwargs['undict']
			if args in bidict and args[0] in undict:
				return float(bidict[args] - discount_value) / undict[args[0]]
			else:
				return 0
		elif len(args) == 3:
			if 'tridict' not in kwargs and 'bidict' not in kwargs:
				raise ValueError('tridict and bidict is required')
			tridict = kwargs['tridict']
			bidict = kwargs['bidict']
			bitup = (args[0], args[1])
			if args in tridict and bitup in bidict:
				return float(tridict[args] - discount_value) / bidict[bitup]
			else:
				return 0
		else:
			return 0
