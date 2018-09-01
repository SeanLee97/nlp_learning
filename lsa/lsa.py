# -*- coding: utf-8 -*-

import numpy as np
import jieba

class LSA(object):
    def __init__(self, docs, kernel=None):
        super(LSA, self).__init__()
        self.kernel = kernel
        self.docs = []
        self.vocabs = set()
        self.build_vocab(docs)

    def tokenizer(self, sent):
        return jieba.lcut(sent)

    def build_vocab(self, docs):
        for doc in docs:
            doc = doc.strip()
            # 为了简单仅仅保留词的长度大于1的
            words = list(filter(lambda x: len(x) > 1, self.tokenizer(doc))) 
            self.docs.append(words)
            self.vocabs.update(words)

        self.vocabs = list(self.vocabs)
        self.word2idx = dict(zip(self.vocabs, range(len(self.vocabs))))

    def build_bow_matrix(self):
        matrix = np.zeros([len(self.vocabs), len(self.docs)])
        for docidx, words in enumerate(self.docs):
            for word in words:
                matrix[self.word2idx[word], docidx] += 1
        return matrix

    def build_tfidf_matrix(self):
        tf = self.build_bow_matrix()
        print(tf)
        df = np.ones([len(self.vocabs), 1])

        for docidx, words in enumerate(self.docs):
            tf[:, docidx] /= np.max(tf[:, docidx])
            for word in words:
                df[self.word2idx[word], 0] += 1
        idf = np.log(len(self.docs)) - np.log(df)

        return tf*idf

    def sim_words(self, k=3):
        if self.kernel == 'tfidf':
            matrix = self.build_tfidf_matrix()
        else:
            matrix = self.build_bow_matrix()

        U, S, Vt = np.linalg.svd(matrix)

        sort_idx = np.argsort(-U)
        # 一般不取第一列，第一列的词往往是本身
        topk = sort_idx[:, 1:k+1] 
        print("word \t similarity")
        for widx, word in enumerate(self.vocabs):
            line = word + ":\t"
            idxs = topk[widx]
            for idx in idxs:
                line += str(self.vocabs[idx]) + " "
            print(line)

    def topic_relate(self, k=2):
        if self.kernel == 'tfidf':
            matrix = self.build_tfidf_matrix()
        else:
            matrix = self.build_bow_matrix()

        U, S, Vt = np.linalg.svd(matrix)

        sort_idx = np.argsort(-Vt, axis=1)
        # 一般不取第一行，第一行是自己本身
        topk = sort_idx[1:k+1, :]
        print(topk)

if __name__ == '__main__':
    doc1 = """计算机科学是系统性研究信息与计算的理论基础以及它们在计算机系统中如何实现与应用的实用技术的学科"""
    
    doc2 = """自然语言处理是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。
    自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""
    
    doc3 = """人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等"""
    
    doc4 = """《瓦尔登湖》是美国作家梭罗独居瓦尔登湖畔的记录，描绘了他两年多时间里的所见、所闻和所思。
    该书崇尚简朴生活，热爱大自然的风光，内容丰厚，意义深远，语言生动"""

    docs = [doc1, doc2, doc3, doc4]

    lsa = LSA(docs, kernel=None)
    lsa.sim_words()
    lsa.topic_relate()