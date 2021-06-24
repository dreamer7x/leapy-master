#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/6 11:57
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot

import numpy
import copy
import leapy


def generate_data():
    # 这个过程就是LDA的生成过程
    # 创建一些数据集
    document_number = 100  # 文本数
    topic_number = 3  # topic数
    word_number = 30  # 单词数
    number_document = numpy.random.randint(150, 200, size=document_number)  # 每个文档的单词数量在150-200
    # 文本主题的狄利克雷参数
    document_topic_hyper = []
    # for _ in range(topic_number):
    #     document_topic_hyper.append(numpy.random.randint(1, topic_number * 10, size=topic_number))
    document_topic_hyper.append(numpy.array((20, 1, 1)))
    document_topic_hyper.append(numpy.array((1, 10, 15)))
    topic_word_hyper = (numpy.ones((topic_number, word_number)) +
                        numpy.array([numpy.arange(word_number) % topic_number == t for t in range(topic_number)]) * 19)
    # 单词话题概率
    topic_word = numpy.array(list(map(lambda x: numpy.random.dirichlet(x), topic_word_hyper)))
    documents = []
    document_topic = numpy.empty((document_number, topic_number))
    # 循环生成d个文档
    for i in range(document_number):
        j = numpy.random.randint(0, topic_number - 1)
        # 生成文档 - 话题概率
        document_topic[i, :] = numpy.random.dirichlet(document_topic_hyper[j], 1)[0]
        # 循环生成每一个文档的单词
        document = []
        for n in range(number_document[i]):
            # 通过 文档 - 话题 概率分布 生成 第d个文档的话题k
            topics = numpy.random.choice(numpy.arange(topic_number), p=document_topic[i, :])
            # 已知话题k 再通过 话题 - 单词 概率分布 生成 单词w
            words = numpy.random.choice(numpy.arange(word_number), p=topic_word[topics, :])
            document.append(words.tolist())
        documents.append(document)
    words = range(word_number)
    return documents, words, topic_number, document_number, word_number, document_topic, topic_word


if __name__ == "__main__":
    model = leapy.LatentDirichletAllocation(debug_enable=True)
    d, w, k, d_n, w_n, d_t, t_w = generate_data()
    model.words = w
    model.k = k
    model.document_number = d_n
    model.word_number = w_n

    model.iteration = 10
    model.gibbs(d)
    document_topic_gibbs = copy.deepcopy(model.document_topic[:, :])

    model.iteration = 5
    model.expectation_maximization(d)
    document_topic_expectation_maximization = copy.deepcopy(model.document_topic[:, :])

    d_t = copy.deepcopy(d_t[:, :])

    print(document_topic_gibbs)
    pyplot.figure(0)
    pyplot.scatter(document_topic_gibbs[:, 0].T.tolist(), document_topic_gibbs[:, 1].T.tolist())

    print(document_topic_expectation_maximization)
    pyplot.figure(1)
    pyplot.scatter(document_topic_expectation_maximization[:, 0].T.tolist(),
                   document_topic_expectation_maximization[:, 1].T.tolist())

    print(d_t)
    pyplot.figure(2)
    pyplot.scatter(d_t[:, 0].T.tolist(), d_t[:, 1].T.tolist())
    pyplot.show()
