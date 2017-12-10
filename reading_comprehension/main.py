# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from dataset import *

import argparse
parser = argparse.ArgumentParser(description='gca main.py')
parser.add_argument('-train', action='store_true', default=False, help='train model')
parser.add_argument('-test', action='store_true', default=False, help='test model')
parser.add_argument('-evaluate', action='store_true', default=False, help='evaluate')
args = parser.parse_args()

def train():
	dataset = BabiDataset(const.task_id)
	model = ReaderNet(len(dataset.QA.VOCAB), const.hidden_size)
	model = load_model(model)
	model = model.cuda() if const.use_cuda else model
	optimizer = optim.Adam(model.parameters(), lr=const.lr_rate)

	best_acc = 0
	for epoch in range(const.epochs):
		model.train()
		total_acc = 0.0
		cnt = 0
		dataset.set_mode('train')
		train_loader = DataLoader(dataset, batch_size=const.batch_size, shuffle=True, collate_fn=pad_collate)
		losses = []
		for batch_idx, data in enumerate(train_loader):
			optimizer.zero_grad()

			contexts, questions, answers = data
			
			contexts = contexts.long()
			contexts = contexts.cuda() if const.use_cuda else contexts
			contexts = Variable(contexts)

			questions = questions.long()
			questions = questions.cuda() if const.use_cuda else questions
			questions = Variable(questions)

			#answers = answers.long()
			answers = answers.cuda() if const.use_cuda else answers
			answers = Variable(answers)

			loss, acc = model.get_loss(contexts, questions, answers)
			losses.append(loss.data[0])
			total_acc += acc * const.batch_size
			if batch_idx % 50 == 0:
				print('loss', np.mean(losses))
				losses = []
			cnt += const.batch_size
			loss.backward()
			optimizer.step()

		dataset.set_mode('valid')
		valid_loader = DataLoader(
			dataset, batch_size=const.batch_size, shuffle=False, collate_fn=pad_collate
		)

		model.eval()
		total_acc = 0.0
		cnt = 0
		for batch_idx, data in enumerate(valid_loader):
			contexts, questions, answers = data
			batch_size = contexts.size()[0]

			contexts = contexts.long()
			contexts = contexts.cuda() if const.use_cuda else contexts
			contexts = Variable(contexts)

			questions = questions.long()
			questions = questions.cuda() if const.use_cuda else questions
			questions = Variable(questions)

			answers = answers.cuda() if const.use_cuda else answers
			answers = Variable(answers)

			_, acc = model.get_loss(contexts, questions, answers)
			total_acc += acc * const.batch_size
			cnt += const.batch_size

		total_acc = total_acc / cnt
		print('accuracy: %.4f' % total_acc)
		if total_acc > best_acc:
			best_acc = total_acc
			best_state = model.state_dict()
			save_model(model)
			print('save model')

def evaluate():
	dataset = BabiDataset(const.task_id)
	model = ReaderNet(len(dataset.QA.VOCAB), const.hidden_size)
	model = load_model(model)
	model = model.cuda() if const.use_cuda else model

	model.eval()
	dataset.set_mode('test')
	test_loader = DataLoader(
		dataset, batch_size=1, shuffle=True, collate_fn=pad_collate
	)
	for batch_idx, data in enumerate(test_loader):
		contexts, questions, answers = data

		print(contexts.size())
		print(questions.size())
		print(answers.size())
		contexts_raw = []
		for cont in contexts.numpy().tolist()[0]:
			c = []
			[c.append(dataset.QA.IVOCAB[w]) for w in cont]
			contexts_raw.append(c)

		q_raw = []
		[q_raw.append(dataset.QA.IVOCAB[w]) for w in questions.numpy().tolist()[0]]

		a_raw = dataset.QA.IVOCAB[answers.numpy().tolist()[0]]

		print('\n>facts: ')
		for cont in contexts_raw:
			print(cont)

		contexts = contexts.long()
		contexts = contexts.cuda() if const.use_cuda else contexts
		contexts = Variable(contexts)
		while True:
			question = input('\n>input your question: ')
			questions = list(map(lambda w: dataset.QA.VOCAB[w] if w in dataset.QA.VOCAB else dataset.QA.VOCAB[const.pad], question.split(' ')))

			#print(questions)

			questions = torch.LongTensor(questions)
			questions = questions.cuda() if const.use_cuda else questions
			questions = Variable(questions).unsqueeze(0)

			pred = model.predict(contexts, questions)
			print(">pred: ", dataset.QA.IVOCAB[pred])

		break;
	pass

if args.train:
	train()
elif args.evaluate:
	evaluate()