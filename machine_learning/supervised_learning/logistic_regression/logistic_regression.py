#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/24 13:24
# @Author   : Mr. Fan

"""
罗杰斯蒂回归
"""

import numpy
import math
import datetime


class LogisticRegressionClassifier:
    def __init__(self, debug_enable=False,
                 algorithm=None,
                 max_epoch=3,
                 max_iteration=300,
                 learning_rate=0.01,
                 seed=100,
                 label_classify=None):
        """
        Parameters
        ----------
        debug_enable : bool
            Whether to enter the debugging state, enter the feedback process information.
            是否进入调试状态，进入将反馈过程信息

        algorithm : str
            To specify the algorithm, which supports:
            1. batch_gradient_descent or BGD
                math:
                    w = w - \alpha \frac{x_i(x_i w - y_i)}{N}
            2. stochastic_gradient_descent or SGD
                math:
                    Same as batch_gradient_descent
            3. mini_batch_gradient_descent or MBGD
                math:
                    Same as batch_gradient_descent

            用以指定算法，支持：
            1. 批量梯度下降法 (batch_gradient_descent or BGD)
            2. 随机梯度下降法 (stochastic_gradient_descent or SGD)
            3. 小批量梯度下降法 (mini_batch_gradient_descent or MBGD)

        label_classify : int
            分类标签

        max_iteration : int 迭代次数
        learning_rate : float 学习率
        seed : int 随机种子
        """
        # 调试
        self.debug_enable = debug_enable

        # 模型参数
        self.w = None
        # 模型偏移
        self.b = None
        # 迭代纪元上限
        self.max_epoch = max_epoch
        # 迭代次数上限
        self.max_iteration = max_iteration
        # 学习率
        self.learning_rate = learning_rate
        # 随机种子
        self.seed = seed
        numpy.random.seed(seed)
        # 分类标签
        self.label_classify = label_classify
        # 算法
        if algorithm is None:
            self.algorithm = "stochastic_gradient_descent"
        else:
            self.algorithm = algorithm

    @staticmethod
    def classify_label(label, label_classify):
        """
        Parameters
        ----------
        label : list 标签列表
        label_classify : int 分类标签

        Returns
        -------
        label : object numpy.matrix shape of N * 1
        """
        label = [1 if i == label_classify else 0 for i in label]
        return numpy.mat(label).T

    def fit(self, sample, label):
        # --------------------------------------------------------------------------------------------------------------
        # 初始化
        # --------------------------------------------------------------------------------------------------------------
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print('------------------------------------------------')
            print('* Error from Classification.fit()             \n'
                  '  The number of samples and tags are different. ')
            print('------------------------------------------------')
            return
        # 如果分类标签非空
        if self.label_classify is not None:
            label = self.classify_label(label, self.label_classify)

        # --------------------------------------------------------------------------------------------------------------
        # 拟合
        # --------------------------------------------------------------------------------------------------------------
        print('[%s] Start training' % datetime.datetime.now())
        if self.debug_enable:
            print("  number of sample: %d" % raw)

        if self.algorithm == "stochastic_gradient_descent":
            self.w, self.b = self.stochastic_gradient_descent(sample, label)
        elif self.algorithm == "batch_gradient_descent":
            self.w, self.b = self.batch_gradient_descent(sample, label)
        elif self.algorithm == "mini_batch_gradient_descent":
            self.w, self.b = self.mini_batch_gradient_descent(sample, label)
        else:
            self.w, self.b = self.stochastic_gradient_descent(sample, label)

    @staticmethod
    def calculate_logistic(X):
        def c(x):
            if x > 300:
                return 1
            if x < -300:
                return 0
            return 1 / (1 + math.exp(-x))

        if isinstance(X, float):
            return c(X)

        elif isinstance(X, numpy.matrix):
            calculator = numpy.vectorize(c)
            X = calculator(X)
            return X

    def calculate_logistic_loss(self, sample, label):
        """
        Notes
        -----
        h 表示罗杰斯蒂函数

        Parameters
        ----------
        sample
        label

        Returns
        -------

        """
        raw, col = sample.shape
        h = self.calculate_logistic(sample * self.w)
        numpy.log()

    # @staticmethod
    # def zero_one(x):
    #     if x > 0:
    #         return 1
    #     else:
    #         return -1

    # m, n = shape(self.dataMat)
    # alpha = 0.001
    # maxCycles = 500
    # weights = ones((n, 1))
    # for k in range(maxCycles):  # heavy on matrix operations
    #     h = sigmoid(self.dataMat * weights)  # matrix mult
    #     error = (self.labelMat - h)  # vector subtraction
    #     weights += alpha * self.dataMat.transpose() * error  # matrix mult
    # return weights

    def batch_gradient_descent(self, sample, label):
        raw, col = sample.shape

        epoch = 0
        while True:
            epoch += 1

            accumulate = numpy.zeros((col, 1))
            for i in range(raw):
                error =


            if epoch >= self.max_epoch:
        for i in range()
        return None

    def mini_batch_gradient_descent(self, sample, label):


    def stochastic_gradient_descent(self, sample, label):
        raw, col = sample.shape
        # w初始化为全1向量
        w = numpy.ones((col, 1), dtype=numpy.float32)
        b = 0

        epoch = 0
        while True:
            epoch += 1

            shuffled_i = numpy.random.permutation(raw)

            iteration = 0
            for i in shuffled_i.tolist():
                iteration += 1
                result = self.calculate_logistic(sample[i, :] * w)
                # 损失
                error = label[i, 0] - result
                # 梯度下降法优化 与感知机类似
                w += self.learning_rate * error * sample[i, :].T
                b += self.learning_rate * error

                if self.debug_enable:


                if iteration >= self.max_iteration:
                    break

            if epoch >= self.max_epoch:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  w: \n\n' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]), end='\n\n')
                    print('  b: ' + str(b))
                break

        return w, b

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            label[i, 0] = 1 if sample[i, :] * self.w + self.b > 0 else 0
        return label

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print('------------------------------------------------')
            print('* Error from Classification.fit()             \n'
                  '  The number of samples and tags are different. ')
            print('------------------------------------------------')
            return

        label_predict = self.predict(sample)
        right_count = 0
        for i in range(raw):
            if label_predict[i, 0] == label[i, 0]:
                right_count += 1
        return right_count / raw
