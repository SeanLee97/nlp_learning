# -*- coding: utf-8 -*-

import math

from collections import defaultdict

class NBayes(object):
    def __init__(self, trainSet):
        self.data = trainSet
        self.tags = defaultdict(int)
        self.tagwords = defaultdict(int)
        self.total = 0

    def _tokenizer(self, sent):
        return list(sent)

    def train(self):
        for tag, doc in self.data:
            words = self._tokenizer(doc)  
            for word in words:
                self.tags[tag] += 1
                self.tagwords[(tag, word)] += 1
                self.total += 1

    def predict(self, inp):
        words = self._tokenizer(inp)

        tmp = {}
        for tag in self.tags.keys():
            tmp[tag] = math.log(self.tags[tag]) -  math.log(self.total)
            for word in words:
                tmp[tag] += math.log(self.tagwords.get((tag, word), 1.0)) - math.log(self.tags[tag])
        ret, score = 0, 0.0
        for t in self.tags.keys():
            cnt = 0.0
            for tt in self.tags.keys():
                cnt += math.exp(tmp[tt] - tmp[t])
            cnt = 1.0 / cnt
            if cnt > score:
                ret, score = t, cnt
        return ret, score



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
                ("neg", "so bad"),
                ("non", "一般般吧，还过的去"), 
                ("non", "不算太好，也不算太差"), 
                ("non", "继续努力吧")
               ]
    clf = NBayes(trainSet)
    clf.train()
    print(clf.predict("不错哦"))
