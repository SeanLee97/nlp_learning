# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np
import const

class adict(dict):
	def __init__(self, *av, **kav):
		dict.__init__(self, *av, **kav)
		self.__dict__ = self

def pad_collate(batch):
	max_context_sen_len = float('-inf')
	max_context_len = float('-inf')
	max_question_len = float('-inf')
	for elem in batch:
		context, question, _ = elem
		max_context_len = max_context_len if max_context_len > len(context) else len(context)
		max_question_len = max_question_len if max_question_len > len(question) else len(question)
		for sen in context:
			max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
	max_context_len = min(max_context_len, 70)
	for i, elem in enumerate(batch):
		_context, question, answer = elem
		_context = _context[-max_context_len:]
		context = np.zeros((max_context_len, max_context_sen_len))
		for j, sen in enumerate(_context):
			context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
		question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
		batch[i] = (context, question, answer)
	return default_collate(batch)

class BabiDataset(Dataset):
	def __init__(self, task_id, mode='train'):
		self.vocab_path = 'dataset/babi{}_vocab.pkl'.format(task_id)
		self.mode = mode
		raw_train, raw_test = get_raw_babi(task_id)
		self.QA = adict()
		self.QA.VOCAB = {const.pad: 0, const.eos: 1}
		self.QA.IVOCAB = {0: const.pad, 1: const.eos}
		self.train = self.get_indexed_qa(raw_train)
		self.valid = [self.train[i][int(-len(self.train[i])/10):] for i in range(3)]
		self.train = [self.train[i][:int(9 * len(self.train[i])/10)] for i in range(3)]
		self.test = self.get_indexed_qa(raw_test)

	def set_mode(self, mode):
		self.mode = mode

	def __len__(self):
		if self.mode == 'train':
			return len(self.train[0])
		elif self.mode == 'valid':
			return len(self.valid[0])
		elif self.mode == 'test':
			return len(self.test[0])

	def __getitem__(self, index):
		if self.mode == 'train':
			contexts, questions, answers = self.train
		elif self.mode == 'valid':
			contexts, questions, answers = self.valid
		elif self.mode == 'test':
			contexts, questions, answers = self.test
		return contexts[index], questions[index], answers[index]

	def get_indexed_qa(self, raw_babi):
		unindexed = get_unindexed_qa(raw_babi)
		questions = []
		contexts = []
		answers = []
		for qa in unindexed:
			context = [c.lower().split() + [const.eos] for c in qa['C']]

			for con in context:
				for token in con:
					self.build_vocab(token)
			context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
			question = qa['Q'].lower().split() + [const.eos]

			for token in question:
				self.build_vocab(token)
			question = [self.QA.VOCAB[token] for token in question]

			self.build_vocab(qa['A'].lower())
			answer = self.QA.VOCAB[qa['A'].lower()]


			contexts.append(context)
			questions.append(question)
			answers.append(answer)
		return (contexts, questions, answers)

	def build_vocab(self, token):
		if not token in self.QA.VOCAB:
			next_index = len(self.QA.VOCAB)
			self.QA.VOCAB[token] = next_index
			self.QA.IVOCAB[next_index] = token


def get_raw_babi(taskid):
	paths = glob('corpus/bAbI/en-10k/qa{}_*'.format(taskid))
	for path in paths:
		if 'train' in path:
			with open(path, 'r') as fp:
				train = fp.read()
		elif 'test' in path:
			with open(path, 'r') as fp:
				test = fp.read()
	return train, test

def build_vocab(raw_babi):
	lowered = raw_babi.lower()
	tokens = re.findall('[a-zA-Z]+', lowered)
	types = set(tokens)
	return types

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def get_unindexed_qa(raw_babi):
	tasks = []
	task = None
	babi = raw_babi.strip().split('\n')
	for i, line in enumerate(babi):
		id = int(line[0:line.find(' ')])
		if id == 1:
			task = {"C": "", "Q": "", "A": "", "S": ""}
			counter = 0
			id_map = {}

		line = line.strip()
		line = line.replace('.', ' . ')
		line = line[line.find(' ')+1:]
		# if not a question
		if line.find('?') == -1:
			task["C"] += line + '<line>'
			id_map[id] = counter
			counter += 1
		else:
			idx = line.find('?')
			tmp = line[idx+1:].split('\t')
			task["Q"] = line[:idx]
			task["A"] = tmp[1].strip()
			task["S"] = [] # Supporting facts
			for num in tmp[2].split():
				task["S"].append(id_map[int(num.strip())])
			tc = task.copy()
			tc['C'] = tc['C'].split('<line>')[:-1]
			tasks.append(tc)
	return tasks

if __name__ == '__main__':
	dset_train = BabiDataset(20, mode='train')
	train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
	for batch_idx, data in enumerate(train_loader):
		contexts, questions, answers = data
		break