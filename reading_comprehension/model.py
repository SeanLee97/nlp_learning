# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------#
# const.python 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import os
import const
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

def save_model(model, path=f'models/model.pth'):
	os.makedirs('models', exist_ok=True)
	with open(path, 'wb') as fp:
		torch.save(model.state_dict(), fp)

def load_model(model, path=f'models/model.pth'):
	if not os.path.exists(path):
		return model
	model.load_state_dict(torch.load(path))
	return model

def position_encoding(embedded_sentence):
	'''
	embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
	l.size() -> (#sentence, #embedding)
	output.size() -> (#batch, #sentence, #embedding)
	'''
	_, _, slen, elen = embedded_sentence.size()

	l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
	l = torch.FloatTensor(l)
	l = l.unsqueeze(0) # for #batch
	l = l.unsqueeze(1) # for #sen
	l = l.expand_as(embedded_sentence)
	l = l.cuda() if const.use_cuda else l
	weighted = embedded_sentence * Variable(l)
	return torch.sum(weighted, dim=2).squeeze(2) # sum with tokens

class InputNet(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(InputNet, self).__init__()
		self.hidden_size = hidden_size
		self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
		for name, param in self.gru.state_dict().items():
			if 'weight' in name: init.xavier_normal(param)
		self.dropout = nn.Dropout(0.1)

	def forward(self, contexts, embedding):
		'''
		contexts.size() -> (#batch, #sentence, #token)
		embedding() -> (#batch, #sentence x #token, #embedding)
		position_encoding() -> (#batch, #sentence, #embedding)
		facts.size() -> (#batch, #sentence, #hidden = #embedding)
		'''
		batch_size, sen_size, token_size = contexts.size()

		contexts = contexts.view(batch_size, -1)
		contexts = embedding(contexts)

		contexts = contexts.view(batch_size, sen_size, token_size, -1)
		contexts = position_encoding(contexts)
		contexts = self.dropout(contexts)

		# init hidden
		h0 = torch.zeros(2, batch_size, self.hidden_size)
		h0 = h0.cuda() if const.use_cuda else h0
		h0 = Variable(h0)

		facts, hdn = self.gru(contexts, h0)
		facts = facts[:, :, :self.hidden_size] + facts[:, :, self.hidden_size:]
		return facts

class QuestionNet(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(QuestionNet, self).__init__()
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

	def forward(self, questions, embedding):
		'''
		questions.size() -> (#batch, #token)
		embedding() -> (#batch, #token, #embedding)
		gru() -> (1, #batch, #hidden)
		'''
		questions = embedding(questions)
		_, questions = self.gru(questions)    # last hidden as questions, (num_layers * num_directions, B, hidden_size)

		questions = questions.transpose(0, 1) # B x 1 x hidden_size
		return questions

class AttnNet(nn.Module):
	def __init__(self, hidden_size):
		super(AttnNet, self).__init__()
		self.hidden_size = hidden_size

	def forward(self, questions, facts):
		batch_size, seqnum, _ = facts.size()

		attn_energies = Variable(torch.zeros(batch_size, seqnum)) # B x S
		for b in range(batch_size):
			for i in range(seqnum):
				attn_energies[b, i] = self.score(facts[b, i], questions[b])  # calc Ct

		attn_energies = attn_energies.cuda() if const.use_cuda else attn_energies
		return F.softmax(attn_energies.unsqueeze(1))
	
	def score(self, fact, question):
		energy = fact.dot(question)
		return energy

class ReaderNet(nn.Module):
	def __init__(self, input_size, hidden_size, dropout_p=0.1):
		super(ReaderNet, self).__init__()

		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.embedding = self.embedding.cuda() if const.use_cuda else self.embedding
		init.uniform(self.embedding.state_dict()['weight'], a=-1.0, b=1.0)

		self.input_net = InputNet(input_size, hidden_size)
		self.question_net = QuestionNet(input_size, hidden_size)
		self.attn_net = AttnNet(hidden_size)
		self.h2o = nn.Linear(hidden_size, input_size)

		self.criterion = nn.CrossEntropyLoss()

	def forward(self, contexts, questions):
		facts = self.input_net(contexts, self.embedding)
		questions = self.question_net(questions, self.embedding).squeeze(1)

		facts_attn = self.attn_net(questions, facts)
		facts = torch.bmm(facts_attn, facts).squeeze(1)
		
		outputs = questions * facts
		outputs = self.h2o(F.tanh(outputs))
		return outputs

	def get_loss(self, contexts, questions, targets):
		output = self.forward(contexts, questions)
		loss = self.criterion(output.view(targets.size(0), -1), targets)
		reg_loss = 0
		for param in self.parameters():
			reg_loss += 0.001 * torch.sum(param * param)
		preds = F.softmax(output)
		_, pred_ids = torch.max(preds, dim=1)
		corrects = (pred_ids.data == targets.data)
		acc = torch.mean(corrects.float())
		return loss + reg_loss, acc		

	def predict(self, contexts, questions):
		output = self.forward(contexts, questions)
		preds = F.softmax(output)
		_, pred_ids = torch.max(preds, dim=1)
		pred_value, pred_ids = torch.topk(preds, 1)
		return pred_ids.data.tolist()[0][0]