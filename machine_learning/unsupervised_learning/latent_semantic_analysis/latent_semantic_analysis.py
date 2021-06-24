#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/1 23:17
# @Author   : Mr. Fan

"""
潜在语义分析
"""

import numpy
import re
import leapy


default_stop_words = {
    "english": ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                "won't", 'wouldn', "wouldn't"]}


class LatentSemanticAnalysis:
    def __init__(self, debug_enable=False):
        self.debug_enable = debug_enable

        # 语种
        self.language = None
        # 话题数
        self.k = None
        # 单词 - 文本矩阵
        self.word_document = None
        # 单词 - 话题矩阵
        self.word_topic = None
        # 话题 - 文本矩阵
        self.topic_document = None
        # 停用词
        self.stop_words = None
        # 单词统计列表
        self.word_map = {}
        # 单词集
        self.words = []
        # 文本数
        self.document_number = 0

    def analysis(self, documents):
        if self.language == 'english':
            for document in documents:
                words = document.lower()
                words = re.sub("[^a-z]", " ", words)
                words = re.split("\\s+", words)

                for word in words:
                    word = word.lower()
                    if word in self.stop_words:
                        continue
                    elif word in self.word_map:
                        self.word_map[word].append(self.document_number)
                    else:
                        self.word_map[word] = [self.document_number]

                self.document_number += 1

            self.words = [k for k in self.word_map.keys() if len(self.word_map[k]) > 1]
            self.words.sort()
            self.word_document = numpy.zeros((len(self.words), self.document_number))
            for i, k in enumerate(self.words):
                for j in self.word_map[k]:
                    self.word_document[i, j] += 1

    def generate(self, documents, k=None, language='english', stop_words=None):
        if stop_words is None:
            stop_words = default_stop_words[language]

        self.stop_words = stop_words
        self.language = language

        self.analysis(documents)

        model = leapy.SingularValueDecomposition(debug_enable=True)
        u, s, v = model.generate(self.word_document)

        if k is None:
            self.k = u.shape[1]
        elif k < u.shape[1]:
            self.k = k
        else:
            self.k = u.shape[1]

        self.word_topic = u[:, :k]
        self.topic_document = s[:k, :k] @ v[:k, :]

    def visualizing(self):
        from matplotlib import pyplot

        pyplot.figure(0)
        pyplot.title("latent_semantic_analysis")
        pyplot.xlabel('dimension_0')
        pyplot.ylabel('dimension_1')

        print(self.topic_document.shape)
        topic_document = self.topic_document.T
        topic_0 = topic_document[0, :]
        topic_1 = topic_document[1, :]
        for i in range(len(topic_0)):
            pyplot.text(topic_0[i], topic_1[i], "topic_" + str(i))
        pyplot.plot(topic_0, topic_1, '.')

        word_topic = self.word_topic.T
        dimension_0 = word_topic[0, :]
        dimension_1 = word_topic[1, :]
        for i in range(word_topic.shape[1]):
            pyplot.text(dimension_0[i], dimension_1[i], self.words[i])
        pyplot.plot(dimension_0, dimension_1, '.')
        pyplot.show()
