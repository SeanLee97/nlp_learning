# !/usr/bin/env python2
# -*- coding: utf-8 -*-

###
# Linux下GB*转UTF-8
###
fin = open('articles.txt', 'r')  
fou = open('articles_uft8.txt', 'w')  
line = fin.readline()  
while line:
    newline = line.decode('GB18030').encode('utf-8')  #用GBK、GB2312都会出错  
    print newline,
    print >> fou, newline,  
    line = fin.readline()  
fin.close()  
fou.close()  
