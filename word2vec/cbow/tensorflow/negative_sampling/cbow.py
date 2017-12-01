# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# skipgram	 			                     #
# author: sean lee                           #
# locate: Shanxi university, Taiyuan, China  #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import numpy as np
import math
import tensorflow as tf 

class Cbow(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def test(self, word, k=10):
		Weight = tf.Variable(tf.random_normal([self.corpus.n_words, const.EMBEDDING_SIZE], -1.0, 1.0))
		inputs = tf.placeholder(tf.int32, [None])
		embed = tf.nn.embedding_lookup(Weight, inputs)

		# cosine
		test_embed = tf.placeholder(tf.float32, [None])
		test_input = tf.placeholder(tf.float32, [None])
		normed_embed = tf.nn.l2_normalize(test_embed, dim=0)
		normed_array = tf.nn.l2_normalize(test_input, dim=0)
		cosine_similarity = tf.reduce_sum(tf.multiply(normed_array, normed_embed))

		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			#restore model
			tf.train.Saver().restore(sess, const.MODEL_PATH)

			vectors = sess.run(embed, feed_dict={inputs: range(self.corpus.n_words)})
			vocab = self.corpus.vocab
			idx = self.corpus.var_word(word)
			scores = []
			for i in range(len(vocab)):
				if vocab[i] == word or vocab[i] == const.U_TOKEN: 
					continue
				vec_a = vectors[i].reshape([-1])
				vec_b = vectors[idx].reshape([-1])
				cosine_sim = sess.run(cosine_similarity, feed_dict={test_embed: vec_a, test_input: vec_b})
				scores.append([vocab[i], cosine_sim]) #calculates cosine similarity
			return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

	def train(self):
		Weight = tf.Variable(tf.truncated_normal([self.corpus.n_words, const.EMBEDDING_SIZE], stddev=1.0/math.sqrt(const.EMBEDDING_SIZE)))
		bias = tf.Variable(tf.zeros([self.corpus.n_words]))

		inputs = tf.placeholder(tf.int32, [const.BATCH_SIZE, const.WIN_SIZE])
		outputs = tf.placeholder(tf.int32, [const.BATCH_SIZE, 1])
		embed = tf.nn.embedding_lookup(tf.random_normal([self.corpus.n_words, const.EMBEDDING_SIZE], -1.0, 1.0), inputs)

		embed_sum = tf.reduce_sum(embed, 1)
		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(Weight, bias, outputs, embed_sum, 3, self.corpus.n_words)) # negative sampling
		optimizer = tf.train.AdamOptimizer(learning_rate=const.LR_RATE).minimize(loss)

		saver = tf.train.Saver()

		losses = []
		with tf.Session() as sess:
			tf.global_variables_initializer().run()

			for epoch in range(const.EPOCH):
				inps, targets = self.corpus.batch_data()
				_, _loss = sess.run([optimizer, loss], feed_dict={inputs:inps, outputs:targets})

				losses.append(_loss)
				if epoch % 100 == 0:
					print('epoch, ', epoch, 'mean loss', np.mean(losses))
					losses= []

			# save model
			saver.save(sess, const.MODEL_PATH)