#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/4 19:16
# @Author   : Mr. Fan

"""

"""

import numpy
import re
import jieba


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


class ProbabilisticLatentSemanticAnalysis:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 语种
        self.language = None
        # 迭代次数
        self.iteration = 0
        # 分词
        self.cut_enable = False
        # 话题数
        self.k = 0
        # 单词 - 文本 概率分布 经验分布
        self.word_document = None
        # 单词集合
        self.words = None
        # 单词 - 话题 概率分布
        self.word_topic = None
        # 话题 - 文本 概率分布
        self.topic_document = None
        # 单词数量
        self.word_number = 0
        # 文本数量
        self.document_number = 0
        # 单词映射
        self.word_map = {}
        # 停用词
        self.stop_words = []
        # 忽略词
        self.ignore_chars = []

    def analysis(self, documents):
        if self.language == 'english':
            for document in documents:
                words = document.lower()
                words = re.sub("[^a-z\\d]", " ", words)
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
            self.word_number = len(self.words)
            self.word_document = numpy.zeros((len(self.words), self.document_number))
            for i, k in enumerate(self.words):
                for j in self.word_map[k]:
                    self.word_document[i, j] += 1

        if self.language == 'chinese':
            for document in documents:
                words = jieba.lcut(document, cut_all=True)

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
            self.word_number = len(self.words)
            self.word_document = numpy.zeros((len(self.words), self.document_number))
            for i, k in enumerate(self.words):
                for j in self.word_map[k]:
                    self.word_document[i, j] += 1

    def generate(self, documents, language='english', algorithm=None, k=2,
                 iteration=10, stop_words=None):
        if stop_words is None:
            stop_words = default_stop_words[language]

        self.stop_words = stop_words
        self.language = language
        self.k = k
        self.iteration = iteration

        self.analysis(documents)

        if algorithm is None:
            algorithm = 'expectation_maximization'

        if algorithm == 'expectation_maximization':
            self.expectation_maximization()

    def expectation_step(self, topic_word_document):
        for i in range(self.word_number):
            for j in range(self.document_number):
                topic_word_document[i, j] = numpy.array([self.word_topic[i] * self.topic_document[:, j]]) / \
                                            numpy.sum([self.word_topic[i] * self.topic_document[:, j]])

    def maximization_step(self, topic_word_document):
        for k in range(self.k):
            for i in range(self.word_number):
                self.word_topic[i, k] = numpy.sum(self.word_document[i] * topic_word_document[i, :, k]) / \
                                        numpy.sum(self.word_document * topic_word_document[:, :, k])

            for j in range(self.document_number):
                self.topic_document[k, j] = numpy.sum(self.word_document[:, j] * topic_word_document[:, j, k]) / \
                                            numpy.sum(self.word_document[:, j])

    def expectation_maximization(self):
        self.word_topic = numpy.random.random((self.word_number, self.k))
        self.topic_document = numpy.random.random((self.k, self.document_number))
        topic_word_document = numpy.zeros((self.word_number, self.document_number, self.k))

        for iteration in range(self.iteration):
            self.expectation_step(topic_word_document)
            self.maximization_step(topic_word_document)

            if self.debug_enable:
                print("报告: ProbabilisticLatentSemanticAnalysis expectation_maximization 单词迭代")
                print("     iteration: " + str(iteration))
