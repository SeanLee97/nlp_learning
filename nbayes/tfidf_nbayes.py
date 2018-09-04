# -*- coding: utf-8 -*-

import jieba
import numpy as np
from collections import defaultdict

class Corpus(object):
    def __init__(self):
        self.word2idx = {}
        self.tags = defaultdict(int)
        self.docs = []
        self.total = 0

    # 分词器
    def tokenizer(self, sent):
        return jieba.lcut(sent)

    # 构建字典，获取分类标记集
    def process_data(self, docs):
        vocabs = set()
        for tag, doc in docs:
            words = self.tokenizer(doc)
            if len(words) == 0:
                continue
            self.tags[tag] += 1
            self.total += 1
            self.docs.append((tag, words))
            vocabs.update(words)
        vocabs = list(vocabs)
        self.word2idx = dict(zip(vocabs, range(len(vocabs))))

    # 计算词袋模型
    def calc_bow(self):
        bow = np.zeros([self.total, len(self.word2idx)])

        for docidx, (tag, doc) in enumerate(self.docs):
            for word in doc:
                bow[docidx, self.word2idx[word]] += 1
        return bow 

    # 计算tf-idf
    def calc_tfidf(self):
        tf = self.calc_bow()
        df = np.ones([1, len(self.word2idx)])

        for docidx, (tag, doc) in enumerate(self.docs):
            tf[docidx] /= np.max(tf[docidx])
            for word in doc:
                df[0, self.word2idx[word]] += 1
        idf = np.log(float(self.total)) - np.log(df)
        return np.multiply(tf, idf)

    # 计算输入词的向量
    def get_vec(self, words):
        vec = np.zeros([1, len(self.word2idx)])
        for word in words:
            if word in self.word2idx:
                vec[0, self.word2idx[word]] += 1
        return vec

class NBayes(Corpus):
    def __init__(self, docs, kernel='tfidf'):
        super(NBayes, self).__init__()
        self.kernel = kernel
        self.process_data(docs)
        self.y_prob = {}
        self.c_prob = None

    def train(self):
        if self.kernel == 'tfidf':
            self.feature = self.calc_tfidf()
        else:
            self.feature = self.calc_bow()

        # 采用极大似然估计计算p(y)
        for tag in self.tags:
            self.y_prob[tag] = float(self.tags[tag]) / self.total

        # 计算条件概率 p(x|y_i)
        self.c_prob = np.zeros([len(self.tags), len(self.word2idx)])
        Z = np.zeros([len(self.tags), 1])
        for docidx in range(len(self.docs)):
            # 获得类别标签id
            tid = self.tags.keys().index(self.docs[docidx][0])
            self.c_prob[tid] += self.feature[docidx]
            Z[tid] = np.sum(self.c_prob[tid])
        self.c_prob /= Z # 归一化

    def predict(self, sent):
        words = self.tokenizer(sent)
        vec = self.get_vec(words)
        ret, max_score = None, -1.0
        for y, pc in zip(self.y_prob, self.c_prob):
            score = np.sum(vec * pc * self.y_prob[y]) # p(x1....xn|yi)p(yi)
            if score > max_score:
                max_score = score
                ret = y
        return ret, 1 - max_score

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
    print(nb.predict("不错哦")) # ('pos', 0.9286)