# -*- coding: utf-8 -*-

import jieba
import numpy as np 

class Sim(object):
    def __init__(self, kernel="tfidf"):
        self.vocabs = set()
        self.kernel = kernel

    def calc_bow(self, docs):
        v = np.zeros([len(docs), len(self.vocabs)])
        for idx, doc in enumerate(docs):
            for word in doc:
                if word in self.vocabs:
                    v[idx, self.vocabs.index(word)] += 1
        return v

    def calc_tfidf(self, docs):
        bow = self.calc_bow(docs)
        tf = np.zeros([2, len(self.vocabs)])
        idf = np.ones([1, len(self.vocabs)])
        for idx, doc in enumerate(docs):
            #tf[idx] = bow[idx] / np.sum(bow[idx])
            tf[idx] = bow[idx] / np.max(bow[idx])
            for word in doc:
                if word in self.vocabs:
                    idf[0, self.vocabs.index(word)] += 1 # 计算词出现在文档中的个数
        return tf * (np.log(len(docs)) - np.log(idf))

    def similarity(self, doc1, doc2):
        l1 = jieba.lcut(doc1)
        l2 = jieba.lcut(doc2)

        self.vocabs.update(l1+l2)
        self.vocabs = list(self.vocabs)

        docs = [l1, l2]

        if self.kernel == 'tfidf':
            vs = self.calc_tfidf(docs)
        else:
            vs = self.calc_bow(docs)

        vsm1, vsm2 = vs[0], vs[1]

        try:
            cos = np.dot(vsm1, vsm2) / (np.linalg.norm(vsm1)*np.linalg.norm(vsm2))
        except:
            cos = None

        return cos

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