# -*- coding: utf-8 -*-

import numpy as np
import jieba

class Sim(object):
    def __init__(self, kernel='tfidf'):
        self.word2idx = {}
        self.kernel = kernel

    def tokenizer(self, sent):
        return jieba.lcut(sent)

    def calc_bow(self, docs):
        bow = np.zeros([len(docs), len(self.word2idx)])
        for docidx, words in enumerate(docs):
            for word in words:
                if word in self.word2idx:
                    bow[docidx, self.word2idx[word]] += 1
        return bow

    def calc_tfidf(self, docs):
        tf = self.calc_bow(docs)
        df = np.ones([1, len(self.word2idx)])

        for docidx, words in enumerate(docs):
            tf[docidx] /= np.max(tf[docidx])
            for word in words:
                if word in self.word2idx:
                    df[0, self.word2idx[word]] += 1
        idf = np.log(len(docs)) - np.log(df)
        tfidf = tf * idf
        return tfidf

    def cos(self, vec1, vec2):
        cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
        try:
            cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
        except:
            cos = None

        return cos

    def similarity(self, doc1, doc2):
        words1 = self.tokenizer(doc1)
        words2 = self.tokenizer(doc2)

        words = set(words1) | set(words2)
        self.word2idx = dict(zip(words, range(len(words))))

        if self.kernel == 'tfidf':
            feature = self.calc_tfidf
        else:
            feature = self.calc_bow

        vec = feature([words1, words2])
        vec1 = vec[0]
        vec2 = vec[1]
        
        return self.cos(vec1, vec2)

if __name__ == '__main__':
    doc1 = """计算机科学（英语：computer science，有时缩写为CS）是系统性研究信息与计算的理论基础以及它们在计算机系统中如何实现与应用的实用技术的学科。
    [1] [2]它通常被形容为对那些创造、描述以及转换信息的算法处理的系统研究。
    计算机科学包含很多分支领域；有些强调特定结果的计算，比如计算机图形学；
    而有些是探讨计算问题的性质，比如计算复杂性理论；还有一些领域专注于怎样实现计算，比如编程语言理论是研究描述计算的方法，
    而程序设计是应用特定的编程语言解决特定的计算问题，人机交互则是专注于怎样使计算机和计算变得有用、好用，以及随时随地为人所用。"""

    doc2 = """自然语言处理（英语：natural language processing，缩写作 NLP）是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""
    sim = Sim()
    print(sim.similarity(doc1, doc2))