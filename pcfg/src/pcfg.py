# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# PCFG Parser	                             #
# author: sean lee                           #
# qq: 929325776							     #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

from collections import defaultdict

class PCFG(object):

	# N_dict - count nonterminal
	# NR_dict - count relation X->Y1 Y2 (X Y1 Y2 are nonterminal)
	# TR_dict - count relation X->y (X is nonterminal y is terminal)
	def __init__(self):
		self.N_dict = defaultdict(int)
		self.NR_dict = defaultdict(int)
		self.TR_dict = defaultdict(int)

	def fit(self, train_corpus):
		with open(train_corpus, 'r') as f:
			for line in f:
				arr = line.strip().split('->')
				self.N_dict[arr[0]] += 1;
				if ' ' in arr[1].strip():
					arr2 = arr[1].split()
					if len(arr2) > 2:
						continue
					self.N_dict[arr2[0]] += 1
					self.N_dict[arr2[1]] += 1
					self.NR_dict[(arr[0], arr2[0], arr2[1])] += 1
				else:
					self.TR_dict[(arr[0], arr[1])] += 1
	# q(X->Y Z)
	def calc_NR_proba(self, x, y1, y2):
		return float(self.NR_dict[(x, y1, y2)]) / self.N_dict[x]

	# q(X->y)
	def calc_TR_proba(self, x, y):
		return float(self.TR_dict[(x, y)]) / self.N_dict[x]

	# Return parse tree
	def parse(self, sentence):
		import json
		print(json.dumps(self.CKY(sentence.split())))

	# CKY algorithm 
	# 适用于CNF (Chomsky normal form)
	def CKY(self, sentence):
		n = len(sentence)
		pi = defaultdict(float) 
		bp = {}	# backpointer
		N = self.N_dict.keys()

		for i in range(n):
			word = sentence[i]
			for X in N:
				pi[(i, i, X)] = self.calc_TR_proba(X, word)

		for i in range(1, n):
			for j in range(n-1):
				k = i + j
				for X in N:
					max_score = 0
					argmax = None
					for R in self.NR_dict.keys():
						if R[0] == X:  # start from X
							Y, Z = R[1:]
							for s in range(j, k):
								if pi[(j, s, Y)] and pi[s+1, k, Z]:
									score = self.calc_NR_proba(X, Y, Z) * pi[(j, s, Y)] * pi[s+1, k, Z]
									if max_score < score:
										max_score = score
										argmax = Y, Z, s
					if max_score:
						pi[j, k, X] = max_score
						bp[j, k, X] = argmax

		# return
		if pi[(0, n-1, 'S')]:
			return self.recover(sentence, bp, 0, n-1, 'S')
		else:
			max_score = 0
			argmax = 0, 0, 'S'
			for X in N:
				if max_score < pi[(0, n-1, X)]:
					max_score = pi[(0, n-1, X)]
					argmax = 0, n-1, X
			return self.recover(sentence, bp, *argmax)

	#  Return the list of the parsed tree with back pointers.
	def recover(self, sentence, bp, i, j, X):
		if i == j:
			return [X, sentence[i]]
		else:
			Y, Z, s = bp[i, j, X]
			return [X, self.recover(sentence, bp, i, s, Y), self.recover(sentence, bp, s+1, j, Z)]