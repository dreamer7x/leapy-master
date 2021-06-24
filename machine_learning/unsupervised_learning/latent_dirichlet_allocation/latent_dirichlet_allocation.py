#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/5 1:55
# @Author   : Mr. Fan

"""

"""

from scipy.special import digamma, polygamma

import numpy
import re
import jieba
import copy


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

default_ignore_chars = {'english': ' ,\'\":!'}


class LatentDirichletAllocation:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        self.k = 0
        self.fit_threshold = 0
        self.language = None
        self.stop_words = None
        self.words = None
        # 单词索引
        self.word_map = []
        # 迭代次数
        self.iteration = 0
        # 单词数
        self.word_number = 0
        # 文本数
        self.document_number = 0

        # 文本 - 话题 概率分布 d - k 服从迪利克雷分布
        self.document_topic = None
        # 文本 - 话题 概率分布 k
        self.document_topic_hyper = None
        # 单词 - 话题 概率分布 k - w 服从迪利克雷分布
        self.topic_word = None
        # 单词 - 话题 迪利克雷分布 超参数 w
        self.topic_word_hyper = None

        self.gibbs_parameters = {
            # 统计数据
            "number_topic_word": None,
            "number_topic": None,
            "number_document_topic": None,
            "number_document": None,
            # 话题序列集合
            "document_topic_sequence": None
        }

        self.expectation_maximization_parameters = {
            # 变分参数
            "document_word_topic": None
        }

    def analysis(self, documents):
        if self.language == 'english':
            clean_documents = []
            words = []
            for document in documents:
                document = document.lower()
                document = re.sub("[^a-z\\d]", " ", document)
                document = re.split("\\s+", document)
                clean_documents.append(document)

                for word in document:
                    if word in self.stop_words:
                        continue
                    if word in words:
                        clean_documents[self.document_number].append(words.index(word))
                    else:
                        words.append(word)
                        clean_documents[self.document_number].append(len(words) - 1)

                self.document_number += 1

            self.words = words
            self.word_number = len(self.words)
            return clean_documents

        if self.language == 'chinese':
            clean_documents = []
            words = []
            for document in documents:
                document = jieba.lcut(document, cut_all=True)

                for word in document:
                    if word in self.stop_words:
                        continue
                    clean_documents[self.document_number].append(word)
                    if word in words:
                        clean_documents[self.document_number].append(words.index(word))
                    else:
                        words.append(word)
                        clean_documents[self.document_number].append(len(words) - 1)

                self.document_number += 1

            self.words = words
            self.word_number = len(self.words)
            return clean_documents

    def generate(self, documents, k=3, language='english',
                 stop_words=None, algorithm=None, iteration=10,
                 fit_threshold=0.1):
        if stop_words is None:
            stop_words = default_stop_words[language]

        self.stop_words = stop_words
        self.document_number = len(documents)
        self.language = language
        self.k = k
        self.iteration = iteration
        self.fit_threshold = fit_threshold

        documents = self.analysis(documents)

        if algorithm is None:
            algorithm = "gibbs"

        if algorithm == "gibbs":
            self.gibbs(documents)

        if algorithm == "expectation_maximization":
            self.expectation_maximization(documents)

    def topic_sequence(self, d, w, documents):
        """
        计算 p(topic | document, word) 的概率值，并通过概率值进行多项分布采样，得到当前的主题

        Parameters
        ----------
        d : int
            文本索引
        w : int
            单词索引
        documents

        Returns
        -------

        """
        # a 话题抽样
        # 减少计数
        old_topic = int(self.gibbs_parameters["document_topic_sequence"][d, w])
        self.gibbs_parameters["number_topic_word"][old_topic][documents[d][w]] -= 1
        self.gibbs_parameters["number_topic"][old_topic] -= 1
        self.gibbs_parameters["number_document_topic"][d][old_topic] -= 1
        self.gibbs_parameters["number_document"][d] -= 1
        # b 按照满条件分布进行抽样
        # 计算单词的话题分布 p(topic | word)
        p = numpy.zeros(self.k)
        for k in range(self.k):
            p[k] = (self.gibbs_parameters["number_document_topic"][d, k] +
                    self.document_topic_hyper[k]) / \
                   (self.gibbs_parameters["number_topic"][k] +
                    numpy.sum(self.document_topic_hyper)) * \
                   (self.gibbs_parameters["number_topic_word"][k, documents[d][w]] +
                    self.topic_word_hyper[documents[d][w]]) / \
                   (self.gibbs_parameters["number_topic"][k] +
                    numpy.sum(self.topic_word_hyper))
        # 归一化处理
        p = p / numpy.sum(p)
        # 抽取主题
        new_topic = numpy.argmax(numpy.random.multinomial(1, p))
        # 更新统计值
        self.gibbs_parameters["number_topic_word"][new_topic][int(documents[d][w])] += 1
        self.gibbs_parameters["number_topic"][new_topic] += 1
        self.gibbs_parameters["number_document_topic"][d][new_topic] += 1
        self.gibbs_parameters["number_document"][d] += 1
        return new_topic

    def gibbs(self, documents):
        self.document_topic_hyper = numpy.ones(self.k)
        self.document_topic = numpy.ones((self.document_number, self.k))
        self.topic_word_hyper = numpy.ones(self.word_number)
        self.topic_word = numpy.ones((self.k, self.word_number))

        # 1 初始化计数
        # 第k个主题 第w个单词 单词数
        self.gibbs_parameters["number_topic_word"] = numpy.zeros((self.k, self.word_number))
        # 第k个主题 单词数
        self.gibbs_parameters["number_topic"] = numpy.zeros(self.k)
        # 第d个文档 第k个主题 单词数
        self.gibbs_parameters["number_document_topic"] = numpy.zeros((self.document_number, self.k))
        # 第d个文档 单词数
        self.gibbs_parameters["number_document"] = numpy.zeros(self.document_number)

        # 2 初始化统计，且对z随机初始化
        document_topic_sequence = numpy.zeros((self.document_number, max([len(i) for i in documents])), dtype=numpy.int)
        for d in range(self.document_number):
            w = len(documents[d])
            for i in range(w):
                rand_topic = int(numpy.random.randint(0, self.k))
                document_topic_sequence[d, i] = rand_topic
                self.gibbs_parameters["number_topic_word"][rand_topic, documents[d][i]] += 1
                self.gibbs_parameters["number_topic"][rand_topic] += 1
                self.gibbs_parameters["number_document_topic"][d, rand_topic] += 1
            self.gibbs_parameters["number_document"][d] = w

        self.gibbs_parameters["document_topic_sequence"] = document_topic_sequence

        # 3 迭代直到进入燃烧期
        iteration = 0
        while True:
            iteration += 1

            # a 指派话题
            for d in range(self.document_number):
                w = len(documents[d])
                for i in range(w):
                    # 更新话题序列
                    topic_sequence = self.topic_sequence(d, i, documents)
                    self.gibbs_parameters["document_topic_sequence"][d, i] = topic_sequence

            if self.debug_enable:
                print("报告: LatentDirichletAllocation gibbs")
                print("     iteration: " + str(iteration))
                print("     document_topic_sequence: " + str(self.gibbs_parameters["document_topic_sequence"]))

            if iteration >= self.iteration:
                print("报告: LatentDirichletAllocation gibbs 迭代完成")
                print("     iteration: " + str(iteration))
                break

        # 4 计算样本模型
        for k in range(self.k):
            for w in range(self.word_number):
                self.topic_word[k, w] = \
                    (self.gibbs_parameters["number_topic_word"][k, w] +
                     self.topic_word_hyper[w]) / \
                    (self.gibbs_parameters["number_topic"][k] +
                     numpy.sum(self.topic_word_hyper))

        for d in range(self.document_number):
            for k in range(self.k):
                self.document_topic[d, k] = \
                    (self.gibbs_parameters["number_document_topic"][d, k] +
                     self.document_topic_hyper[k]) / \
                    (self.gibbs_parameters["number_document"][d] +
                     numpy.sum(self.document_topic_hyper))

    def expectation_maximization(self, documents):
        # 初始化
        self.document_topic_hyper = numpy.ones(self.k)
        self.document_topic = numpy.random.dirichlet(100 * numpy.ones(self.k), self.document_number)
        self.topic_word_hyper = numpy.ones(self.word_number)
        self.topic_word = numpy.random.dirichlet(100 * numpy.ones(self.word_number), self.k)
        self.expectation_maximization_parameters["document_word_topic"] = \
            numpy.array([numpy.random.dirichlet(100 * self.document_topic_hyper, len(i)) for i in documents],
                        dtype=object)

        iteration = 0
        while True:
            iteration += 1

            self.expectation_step(documents)
            self.maximization_step()

            if self.debug_enable:
                print("报告: LatentDirichletAllocation expectation_maximization")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("报告: LatentDirichletAllocation expectation_maximization 迭代完成")
                break

    def expectation_step(self, documents):
        # 更新 document_topic
        self.update_document_topic()
        # 更新 topic_word
        self.update_topic_word(documents)
        # 更新 document_word_topic
        self.update_document_word_topic(documents)

    def maximization_step(self):
        # 更新 document_topic_hyper
        self.update_document_topic_hyper()
        # 更新 topic_word_hyper
        self.update_topic_word_hyper()

    def update_document_topic(self):
        document_topic_hyper = copy.deepcopy(self.document_topic_hyper)
        document_word_topic = copy.deepcopy(self.expectation_maximization_parameters["document_word_topic"])
        # 预定义占位
        document_topic = numpy.zeros((self.document_number, self.k))
        for d in range(self.document_number):
            document_topic[d] = document_topic_hyper + numpy.sum(document_word_topic[d], axis=0)
        # 归一化
        document_topic = numpy.array([document_topic[:, k] /
                                      numpy.sum(document_topic, axis=1)
                                      for k in range(self.k)]).T
        self.document_topic = document_topic

    def update_topic_word(self, documents):
        topic_word_hyper = copy.deepcopy(self.topic_word_hyper)
        document_word_topic = copy.deepcopy(self.expectation_maximization_parameters["document_word_topic"])
        topic_word = numpy.zeros((self.k, self.word_number))

        for k in range(self.k):
            for w in range(self.word_number):
                sum_document_word = 0
                for d in range(self.document_number):
                    for i in range(len(documents[d])):
                        sum_document_word += document_word_topic[d][i][k] * \
                                             (1 if documents[d][i] == self.words[w] else 0)
                topic_word[k][w] = sum_document_word + topic_word_hyper[w]
        # 归一化
        topic_word = numpy.array([topic_word[k] / numpy.sum(topic_word[k]) for k in range(self.k)])
        self.topic_word = topic_word

    def update_document_word_topic(self, documents):
        document_topic = copy.deepcopy(self.document_topic)
        topic_word = copy.deepcopy(self.topic_word)
        document_word_topic = numpy.array([numpy.ones((len(i), self.k)) for i in documents], dtype=object)

        for d in range(self.document_number):
            for i in range(len(documents[d])):
                sum_topic = 0
                document_word_array = numpy.zeros(self.k)
                for k in range(self.k):
                    sum_word_topic = 0
                    for w in range(self.word_number):
                        sum_word_topic += digamma(topic_word[k][w]) * (self.words[w] == documents[d][i])
                    index = digamma(document_topic[d][k]) + sum_word_topic + digamma(numpy.sum(topic_word[k]))
                    # 为防止指数增长带来的内存溢出风险，限定一个阈值
                    if index > 20:
                        index = 20
                    sum_topic += numpy.exp(index)
                    document_word_array[k] = numpy.exp(index)
                # k维数据归一化 赋值
                document_word_topic[d][i, :] = document_word_array / sum_topic
        self.expectation_maximization_parameters["document_word_topic"] = document_word_topic

    def update_document_topic_hyper(self):
        document_topic_hyper = copy.deepcopy(self.document_topic_hyper)
        document_topic = copy.deepcopy(self.document_topic)
        iteration = 0
        while True:
            iteration += 1
            alpha_old = document_topic_hyper

            # 计算 的一阶导数
            # numpy.tile document_topic_hyper 数据扩展
            g = self.document_number * (digamma(numpy.sum(document_topic_hyper)) - digamma(document_topic_hyper)) + \
                numpy.sum(digamma(document_topic) - numpy.tile(digamma(numpy.sum(document_topic, axis=1)),
                                                               (self.k, 1)).T, axis=0)
            # 计算 hessian 矩阵
            h = -1 * self.document_number * polygamma(1, document_topic_hyper)
            z = self.document_number * polygamma(1, numpy.sum(document_topic_hyper))
            c = numpy.sum(g / h) / (z ** (-1.0) + numpy.sum(h ** (-1.0)))
            # 更新 document_topic_hyper
            document_topic_hyper = document_topic_hyper - (g - c) / h
            # 终止阀值
            if numpy.sqrt(numpy.mean(numpy.square(document_topic_hyper - alpha_old))) < self.fit_threshold:
                print("报告: LatentDirichletAllocation update_document_topic_hyper 拟合达到理想状态")
                print("     iteration: " + str(iteration))
                break

            if self.debug_enable:
                print("报告: LatentDirichletAllocation update_document_topic_hyper")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("报告: LatentDirichletAllocation update_document_topic_hyper 迭代完成")
                break
        self.document_topic_hyper = document_topic_hyper

    def update_topic_word_hyper(self):
        topic_word_hyper = copy.deepcopy(self.topic_word_hyper)
        topic_word = copy.deepcopy(self.topic_word)

        iteration = 0
        while True:
            iteration += 1

            old_topic_word_hyper = topic_word_hyper
            g = self.k * (digamma(numpy.sum(topic_word_hyper)) - digamma(topic_word_hyper)) + \
                numpy.sum(digamma(topic_word) - numpy.tile(digamma(numpy.sum(topic_word, axis=1)),
                                                           (self.word_number, 1)).T, axis=0)
            h = -1 * self.k * polygamma(1, topic_word_hyper)
            z = self.k * polygamma(1, numpy.sum(topic_word_hyper))
            c = numpy.sum(g / h) / (z ** (-1.0) + numpy.sum(h ** (-1.0)))
            topic_word_hyper = topic_word_hyper - (g - c) / h
            if numpy.sqrt(numpy.mean(numpy.square(topic_word_hyper - old_topic_word_hyper))) < self.fit_threshold:
                print("报告: LatentDirichletAllocation update_topic_word_hyper 拟合达到理想状态")
                print("     iteration: " + str(iteration))
                break

            if self.debug_enable:
                print("报告: LatentDirichletAllocation update_topic_word_hyper")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("报告: LatentDirichletAllocation update_topic_word_hyper 迭代完成")
                break
        self.topic_word_hyper = topic_word_hyper
