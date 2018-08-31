# -*- coding: utf-8 -*-

import jieba
import numpy as np 
from collections import defaultdict

# bag of word
class Corpus(object):
    def __init__(self, data):
        self.tags = defaultdict(int)
        self.vocabs = set()
        self.docs = []

        self.build_vocab(data)

        self.v_l = len(self.vocabs)
        self.d_l = len(self.docs)


    def tokenizer(self, sent):
        return jieba.lcut(sent)
        #return list(sent)

    def build_vocab(self, data):
        for (tag, doc) in data:
            words = self.tokenizer(doc)
            self.vocabs.update(words)
            self.tags[tag] += 1
            self.docs.append((tag, words))
        self.vocabs = list(self.vocabs)

    def get_idx(self, words):
        bow = np.zeros([1, self.v_l])
        for word in words:
            if word in self.vocabs:
                bow[0, self.vocabs.index(word)] += 1
        return bow

    def calc_bow(self):
        self.bow = np.zeros([self.d_l, self.v_l])
        for idx in range(self.d_l):
            for word in self.docs[idx][1]:
                if word in self.vocabs:
                    self.bow[idx, self.vocabs.index(word)] += 1

    def calc_tfidf(self):
        # calc tf
        self.calc_bow()

        self.tf = np.zeros([self.d_l, self.v_l])
        self.idf = np.ones([1, self.v_l])
        self.tf_idf = np.ones([self.d_l, self.v_l])
        for idx in range(self.d_l):
            self.tf[idx] = self.bow[idx] /np.sum(self.bow[idx])
            for word in self.vocabs:
                self.idf[0, self.vocabs.index(word)] += 1
        self.idf = np.log(float(self.d_l) / self.idf)
        self.tfidf = self.tf * self.idf

class NBayes(Corpus):
    def __init__(self, data, kernel="tfidf"):
        super(NBayes, self).__init__(data)

        self.kernel = kernel
        self.y_prob = {} # p(y_i)
        self.c_prob = None # p(x|y_i) , Condition Proba
        self.feature = None

    def train(self):
        if self.kernel == "tfidf":
            self.calc_tfidf()
            self.feature = self.tfidf
        else:
            self.calc_bow()
            self.feature = self.bow

        for tag in self.tags:
            self.y_prob[tag] = float(self.tags[tag])/ self.d_l

        self.c_prob = np.zeros([len(self.tags), self.v_l])
        Z = np.zeros([len(self.tags), 1])

        for idx in range(self.d_l):
            tid = list(self.tags.keys()).index(self.docs[idx][0])
            self.c_prob[tid] += self.feature[idx]
            Z[tid] = np.sum(self.c_prob[tid])

        self.c_prob /= Z

    def predict(self, inp):
        words = self.tokenizer(inp)
        idx = self.get_idx(words)

        tag, score = None, -1
        for (p_c, y) in zip(self.c_prob, self.y_prob):
            tmp = np.sum(idx * p_c * self.y_prob[y])

            if tmp > score:
                tag = y
                score = tmp
        return tag, 1.0 - score

if __name__ == '__main__':
    trainSet = [("pos", "good job !"),
                ("pos", "表现不错哦"), 
                ("pos", "厉害咯"), 
                ("pos", "做的很好啊"), 
                ("pos", "做得不错继续努力"),
                ("pos", "不错！点赞"),
                ("neg", "太差了"), 
                ("neg", "太糟糕了"), 
                ("neg", "你做的一点都不好"), 
                ("neg", "不行，重做"),
                ("neg", "so bad"),
                ("non", "一般般吧，还过的去"), 
                ("non", "不算太好，也不算太差"), 
                ("non", "继续努力吧")
               ]
               
    nb = NBayes(trainSet)
    nb.train()
    print(nb.predict("不行"))
