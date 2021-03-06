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
        # ??????
        self.debug_enable = debug_enable

        self.k = 0
        self.fit_threshold = 0
        self.language = None
        self.stop_words = None
        self.words = None
        # ????????????
        self.word_map = []
        # ????????????
        self.iteration = 0
        # ?????????
        self.word_number = 0
        # ?????????
        self.document_number = 0

        # ?????? - ?????? ???????????? d - k ????????????????????????
        self.document_topic = None
        # ?????? - ?????? ???????????? k
        self.document_topic_hyper = None
        # ?????? - ?????? ???????????? k - w ????????????????????????
        self.topic_word = None
        # ?????? - ?????? ?????????????????? ????????? w
        self.topic_word_hyper = None

        self.gibbs_parameters = {
            # ????????????
            "number_topic_word": None,
            "number_topic": None,
            "number_document_topic": None,
            "number_document": None,
            # ??????????????????
            "document_topic_sequence": None
        }

        self.expectation_maximization_parameters = {
            # ????????????
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
        ?????? p(topic | document, word) ?????????????????????????????????????????????????????????????????????????????????

        Parameters
        ----------
        d : int
            ????????????
        w : int
            ????????????
        documents

        Returns
        -------

        """
        # a ????????????
        # ????????????
        old_topic = int(self.gibbs_parameters["document_topic_sequence"][d, w])
        self.gibbs_parameters["number_topic_word"][old_topic][documents[d][w]] -= 1
        self.gibbs_parameters["number_topic"][old_topic] -= 1
        self.gibbs_parameters["number_document_topic"][d][old_topic] -= 1
        self.gibbs_parameters["number_document"][d] -= 1
        # b ?????????????????????????????????
        # ??????????????????????????? p(topic | word)
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
        # ???????????????
        p = p / numpy.sum(p)
        # ????????????
        new_topic = numpy.argmax(numpy.random.multinomial(1, p))
        # ???????????????
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

        # 1 ???????????????
        # ???k????????? ???w????????? ?????????
        self.gibbs_parameters["number_topic_word"] = numpy.zeros((self.k, self.word_number))
        # ???k????????? ?????????
        self.gibbs_parameters["number_topic"] = numpy.zeros(self.k)
        # ???d????????? ???k????????? ?????????
        self.gibbs_parameters["number_document_topic"] = numpy.zeros((self.document_number, self.k))
        # ???d????????? ?????????
        self.gibbs_parameters["number_document"] = numpy.zeros(self.document_number)

        # 2 ????????????????????????z???????????????
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

        # 3 ???????????????????????????
        iteration = 0
        while True:
            iteration += 1

            # a ????????????
            for d in range(self.document_number):
                w = len(documents[d])
                for i in range(w):
                    # ??????????????????
                    topic_sequence = self.topic_sequence(d, i, documents)
                    self.gibbs_parameters["document_topic_sequence"][d, i] = topic_sequence

            if self.debug_enable:
                print("??????: LatentDirichletAllocation gibbs")
                print("     iteration: " + str(iteration))
                print("     document_topic_sequence: " + str(self.gibbs_parameters["document_topic_sequence"]))

            if iteration >= self.iteration:
                print("??????: LatentDirichletAllocation gibbs ????????????")
                print("     iteration: " + str(iteration))
                break

        # 4 ??????????????????
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
        # ?????????
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
                print("??????: LatentDirichletAllocation expectation_maximization")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("??????: LatentDirichletAllocation expectation_maximization ????????????")
                break

    def expectation_step(self, documents):
        # ?????? document_topic
        self.update_document_topic()
        # ?????? topic_word
        self.update_topic_word(documents)
        # ?????? document_word_topic
        self.update_document_word_topic(documents)

    def maximization_step(self):
        # ?????? document_topic_hyper
        self.update_document_topic_hyper()
        # ?????? topic_word_hyper
        self.update_topic_word_hyper()

    def update_document_topic(self):
        document_topic_hyper = copy.deepcopy(self.document_topic_hyper)
        document_word_topic = copy.deepcopy(self.expectation_maximization_parameters["document_word_topic"])
        # ???????????????
        document_topic = numpy.zeros((self.document_number, self.k))
        for d in range(self.document_number):
            document_topic[d] = document_topic_hyper + numpy.sum(document_word_topic[d], axis=0)
        # ?????????
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
        # ?????????
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
                    # ?????????????????????????????????????????????????????????????????????
                    if index > 20:
                        index = 20
                    sum_topic += numpy.exp(index)
                    document_word_array[k] = numpy.exp(index)
                # k?????????????????? ??????
                document_word_topic[d][i, :] = document_word_array / sum_topic
        self.expectation_maximization_parameters["document_word_topic"] = document_word_topic

    def update_document_topic_hyper(self):
        document_topic_hyper = copy.deepcopy(self.document_topic_hyper)
        document_topic = copy.deepcopy(self.document_topic)
        iteration = 0
        while True:
            iteration += 1
            alpha_old = document_topic_hyper

            # ?????? ???????????????
            # numpy.tile document_topic_hyper ????????????
            g = self.document_number * (digamma(numpy.sum(document_topic_hyper)) - digamma(document_topic_hyper)) + \
                numpy.sum(digamma(document_topic) - numpy.tile(digamma(numpy.sum(document_topic, axis=1)),
                                                               (self.k, 1)).T, axis=0)
            # ?????? hessian ??????
            h = -1 * self.document_number * polygamma(1, document_topic_hyper)
            z = self.document_number * polygamma(1, numpy.sum(document_topic_hyper))
            c = numpy.sum(g / h) / (z ** (-1.0) + numpy.sum(h ** (-1.0)))
            # ?????? document_topic_hyper
            document_topic_hyper = document_topic_hyper - (g - c) / h
            # ????????????
            if numpy.sqrt(numpy.mean(numpy.square(document_topic_hyper - alpha_old))) < self.fit_threshold:
                print("??????: LatentDirichletAllocation update_document_topic_hyper ????????????????????????")
                print("     iteration: " + str(iteration))
                break

            if self.debug_enable:
                print("??????: LatentDirichletAllocation update_document_topic_hyper")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("??????: LatentDirichletAllocation update_document_topic_hyper ????????????")
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
                print("??????: LatentDirichletAllocation update_topic_word_hyper ????????????????????????")
                print("     iteration: " + str(iteration))
                break

            if self.debug_enable:
                print("??????: LatentDirichletAllocation update_topic_word_hyper")
                print("     iteration: " + str(iteration))

            if iteration >= self.iteration:
                print("??????: LatentDirichletAllocation update_topic_word_hyper ????????????")
                break
        self.topic_word_hyper = topic_word_hyper
