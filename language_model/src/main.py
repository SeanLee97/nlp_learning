# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import load_dataset
from smooth import Smooth
from ngram import *
from evaluate import perplexity

train_dataset = load_dataset('./corpus/toy/train.txt')
test_dataset = load_dataset('./corpus/toy/test.txt')


###################### ungram start ######################

'''
model_unsmooth = UnGram(train_dataset)
model_smooth = UnGram(train_dataset, Smooth.discounting)

vocabs = model_unsmooth.sort_vocab()

print("- ungram unsmooth -")
GramUtil.print_ungram_probs(model_unsmooth, vocabs)

print("- ungram smooth -")
GramUtil.print_ungram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_ungram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_ungram_count))
'''
###################### ungram end ######################


###################### bigram start ######################

model_unsmooth = BiGram(train_dataset)
model_smooth = BiGram(train_dataset, Smooth.discounting)

vocabs = model_unsmooth.sort_vocab()

print("- bigram unsmooth -")
GramUtil.print_bigram_probs(model_unsmooth, vocabs)

print("- bigram smooth -")
GramUtil.print_bigram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_bigram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_bigram_count))

###################### ungram end ######################


###################### trigram start ######################
'''
model_unsmooth = TriGram(train_dataset)
model_smooth = TriGram(train_dataset, Smooth.discounting)

vocabs = model_unsmooth.sort_vocab()

print("- ungram unsmooth -")
GramUtil.print_trigram_probs(model_unsmooth, vocabs)

print("- ungram smooth -")
GramUtil.print_trigram_probs(model_smooth, vocabs)

print('- sentence_prob -')
print("\t\t smooth\t\t unsmooth")
for sentence in test_dataset:
	smooth = "{0:.5f}".format(model_smooth.calc_sentence_prob(sentence))
	unsmooth = "{0:.5f}".format(model_unsmooth.calc_sentence_prob(sentence))
	print("".join(sentence), "\t", smooth, "\t", unsmooth)

print("- test perplexity -")
print("unsmooth: ", perplexity(model_smooth, test_dataset, GramUtil.calc_bigram_count))
print("smooth: ", perplexity(model_unsmooth, test_dataset, GramUtil.calc_bigram_count))
'''
###################### ungram end ######################
