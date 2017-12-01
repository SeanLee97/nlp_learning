# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# skipgram	 			                     #
# author: sean lee                           #
# email: lxm_0828@163.com                    #
#--------------------------------------------#

import const
import math
import numpy as np
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
				scores.append([vocab[i], cosine_sim]) #cosine similarity
			return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

	def train(self):
		Weight = tf.Variable(tf.truncated_normal([self.corpus.n_words, const.EMBEDDING_SIZE], stddev=1.0/math.sqrt(const.EMBEDDING_SIZE)))
		bias = tf.Variable(tf.random_normal([self.corpus.n_words]))

		inputs = tf.placeholder(tf.int32, [const.BATCH_SIZE, const.WIN_SIZE])
		targets = tf.placeholder(tf.int32, [const.BATCH_SIZE, 1])
		vocabs = tf.placeholder(tf.int32, [const.BATCH_SIZE, self.corpus.n_words])

		embed_weight_v = tf.Variable(tf.random_normal([self.corpus.n_words, const.EMBEDDING_SIZE], -1.0, 1.0))
		embed_weight_u = tf.Variable(tf.random_normal([self.corpus.n_words, const.EMBEDDING_SIZE], -1.0, 1.0))
		embed_weight_actual = tf.Variable(tf.random_normal([self.corpus.n_words, const.EMBEDDING_SIZE], -1.0, 1.0))
		embed_v = tf.nn.embedding_lookup(embed_weight_v, inputs)
		embed_u = tf.nn.embedding_lookup(embed_weight_u, targets)
		embed_actual = tf.nn.embedding_lookup(embed_weight_actual, vocabs)

		'''
		print(embed_u.shape)
		print(embed_v.shape)
		print(embed_actual.shape)
		exit()
		'''
		embed_v_trans = tf.transpose(embed_v, [0, 2, 1])

		#print(embed_v_trans.shape)
		scores = tf.matmul(embed_u, embed_v_trans)   		
		norm_scores = tf.matmul(embed_actual, embed_v_trans) 

		softmax = tf.exp(scores) / tf.reduce_sum(tf.exp(norm_scores), 1)
		softmax = tf.expand_dims(softmax, 1)
		nll_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(softmax,1e-10,1.0))) 

		optimizer = tf.train.AdamOptimizer(learning_rate=const.LR_RATE).minimize(nll_loss)

		saver = tf.train.Saver()

		losses = []
		with tf.Session() as sess:
			tf.global_variables_initializer().run()

			for epoch in range(const.EPOCH):
				_inputs, _targets = self.corpus.batch_data()

				#print(_inputs.shape, _targets.shape)
				#continue
				#_inputs = np.hstack(_inputs)   # (2, )
				#_inputs = _inputs.reshape(_inputs.shape[0], 1)
				_targets = np.vstack(_targets) # (2, 1)

				vocab = self.corpus.var_sentence(self.corpus.vocab)
				_vocabs = []
				[_vocabs.append(vocab) for x in range(inputs.shape[0])]
				_vocabs = np.array(_vocabs)

				_, _loss = sess.run([optimizer, nll_loss], feed_dict={inputs:_inputs, targets:_targets, vocabs: _vocabs})
				losses.append(_loss)

				if epoch % 10 == 0:
					print('epoch, ', epoch, 'mean loss', np.mean(losses))
					losses= []

			# save model
			saver.save(sess, const.MODEL_PATH)