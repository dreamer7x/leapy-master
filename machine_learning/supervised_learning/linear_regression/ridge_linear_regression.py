#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 23:42
# @Author   : Mr. Fan

"""
岭回归
"""

import numpy
import itertools
import datetime


class RidgeLinearRegression:

    def __init__(self, debug_enable=False,
                 threshold_enable=True, fit_threshold=0.1,
                 learning_rate=0.01, algorithm=None, max_iteration=100,
                 penalty_factor=0.1, polynomial_degree=0):
        """
        初始化标准线性回归类对象

        Parameters
        ----------
        debug_enable : bool
            Whether to enter the debugging state, enter the feedback process information.
            是否进入调试状态，进入将反馈过程信息

        threshold_enable : bool
            Whether to determine the fitting threshold.
            是否判定拟合阀值

        polynomial_degree : int
            Used to specify polynomial depth. 0 skims
            用以指定多项式深度，为 0 则掠过

        algorithm : str
            To specify the algorithm, which supports:
            1. math_solving : Solved by a mathematical formula.
               Since the regular term constant is added, there is no irreversible risk.
                math:
                    (X^T X + eye(p, \lambda))^{-1} X^T Y
            2. batch_gradient_descent or BGD
                math:
                    w = w - \alpha \frac{x_i(x_i w - y_i) + 2 \lambda w}{N}
            3. stochastic_gradient_descent or SGD
                math:
                    Same as batch_gradient_descent
            4. mini_batch_gradient_descent or MBGD
                math:
                    Same as batch_gradient_descent

            用以指定算法，支持：
            1. 数学求解法 (math_solving) : 通过数学公式求解，由于加上了正则项常数，并不存在不可逆风险
            2. 批量梯度下降法 (batch_gradient_descent or BGD)
            3. 随机梯度下降法 (stochastic_gradient_descent or SGD)
            4. 小批量梯度下降法 (mini_batch_gradient_descent or MBGD)

        fit_threshold : float 拟合阈值
        learning_rate : float 学习率
        max_iteration : int 迭代上限
        penalty_factor : float 惩罚系数
        """
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        # 模型参数
        self.w = None
        # 是否比对阀值
        self.threshold_enable = threshold_enable
        # 拟合阀值
        self.fit_threshold = fit_threshold
        # 学习率
        self.learning_rate = learning_rate
        # 算法
        if algorithm is None:
            self.algorithm = "mini_batch_gradient_descent"
        else:
            self.algorithm = algorithm
        # 迭代上线
        self.max_iteration = max_iteration
        # 惩罚系数
        self.penalty_factor = penalty_factor
        # 多项式转换深度
        self.polynomial_degree = polynomial_degree

    def fit(self, train_sample, train_label):
        """
        拟合

        Parameters
        ----------
        train_sample : numpy.array N * p, numpy.matrix N * p, list N * [p]
            训练样本
        train_label : numpy.array N * 1 / 1 * N, numpy.matrix N * 1 / 1 * N, list N
            训练标签
        """
        # --------------------------------------------------------------------------------------------------------------
        # 初始化
        # --------------------------------------------------------------------------------------------------------------
        train_sample = numpy.mat(train_sample)
        raw, col = train_sample.shape

        train_label = numpy.mat(train_label)
        if train_label.shape[0] == 1:
            train_label = train_label.T
        if train_label.shape[0] != raw:
            print('------------------------------------------------')
            print('* Error from Classification.fit()             \n'
                  '  The number of samples and tags are different  ')
            print('------------------------------------------------')
            return

        # 是否进行多项式转换
        if self.polynomial_degree != 0:
            train_sample = self.polynomial_features(train_sample, self.polynomial_degree)

        # --------------------------------------------------------------------------------------------------------------
        # 拟合
        # --------------------------------------------------------------------------------------------------------------
        print('[%s] Start training' % datetime.datetime.now())

        if self.algorithm == "math_solving":
            self.w = self.math_solving(train_sample, train_label)
        elif self.algorithm == "batch_gradient_descent":
            if self.threshold_enable:
                self.w = self.batch_gradient_descent_with_threshold(train_sample, train_label)
            else:
                self.w = self.batch_gradient_descent(train_sample, train_label)
        elif self.algorithm == "stochastic_gradient_descent":
            self.w = self.stochastic_gradient_descent(train_sample, train_label)
        elif self.algorithm == "mini_batch_gradient_descent":
            if self.threshold_enable:
                self.w = self.mini_batch_gradient_descent_with_threshold(train_sample, train_label)
            else:
                self.w = self.mini_batch_gradient_descent(train_sample, train_label)
        else:
            if self.debug_enable:
                print("* Warning from RidgeLinearRegression.fit() ")
                print("  summary: 算法未指定")
            if self.threshold_enable:
                self.w = self.mini_batch_gradient_descent_with_threshold(train_sample, train_label)
            else:
                self.w = self.mini_batch_gradient_descent(train_sample, train_label)

        print('[%s] Training done' % datetime.datetime.now())

    @staticmethod
    def residuals(sample, label, w):
        """
        根据样本，标签，模型参数计算残差

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量
        w : numpy.matrix p * 1
            模型参数

        Returns
        -------
        float 残差
        """
        residuals = label - sample @ w
        residuals = residuals.T @ residuals
        return residuals[0, 0]

    # 多项式特征转换
    @staticmethod
    def polynomial_features(sample, degree):
        """
        多项式特征转换

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        degree : int
            转换深度

        Returns
        -------
        numpy.matrix 转换后样本
        """
        raw, col = sample.shape

        combinations = [itertools.combinations_with_replacement(range(col), i) for i in range(degree)]
        # [(), (0), (0, 0), (0, 0, 0), (0, 0, 0, 0), ...]
        combinations = [j for i in combinations for j in i]

        new_col = len(combinations)
        # 创建多项式样本集
        new_sample = numpy.mat(numpy.empty((raw, new_col)))
        # 生成多项式样本集
        for i, j in enumerate(combinations):
            new_sample[:, i] = numpy.prod(sample[:, j], axis=1)

        return new_sample

    def math_solving(self, sample, label):
        """
        直接求解法

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量

        Returns
        -------
        numpy.matrix 模型参数 w
        """
        print("* Start math_solving")

        # eye方法生成单位矩阵
        w = sample.T * sample + numpy.eye(sample.shape[1]) * self.penalty_factor
        # 岭线性回归求解公式
        w = w.I * sample.T @ label

        print('* Training finished')
        if self.debug_enable:
            print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
        return w

    def batch_gradient_descent(self, sample, label):
        """
        Parameters
        ----------
        sample : numpy.matrix shape of N * p
        label : numpy.matrix shape of N * 1

        Returns
        -------
        numpy.matrix 模型参数 w
        """
        print('* Start batch_gradient_descent BGD')
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        w = numpy.mat(numpy.zeros((col, 1)))

        iteration = 0
        while True:
            iteration += 1
            for i in range(raw):
                difference = sample[i, :] * w - label[i, 0]
                w -= self.learning_rate * (sample[i, :].reshape(-1, 1) * difference
                                           + 2 * self.penalty_factor * w) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration >= self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach max_iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

        return w

    def batch_gradient_descent_with_threshold(self, sample, label):
        """
        批量梯度下降

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量

        Returns
        -------
        numpy.matrix 模型参数 w
        """
        print('* Start batch_gradient_descent BGD')
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        w = numpy.mat(numpy.zeros((col, 1)))

        iteration = 0
        last_residuals = self.residuals(sample, label, w)
        while True:
            iteration += 1
            for i in range(raw):
                difference = sample[i, :] * w - label[i, 0]
                w -= self.learning_rate * (sample[i, :].reshape(-1, 1) * difference
                                           + 2 * self.penalty_factor * w) / raw

            residuals = self.residuals(sample, label, w)
            if self.debug_enable:
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if abs(residuals - last_residuals) < self.fit_threshold:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach fit_threshold')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            if iteration >= self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach max_iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            last_residuals = residuals

        return w

    def stochastic_gradient_descent(self, sample, label):
        """
        随机梯度下降算法

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量

        Returns
        -------
        numpy.matrix p * 1

        """
        print("* Start stochastic_gradient_descent SGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        w = numpy.mat(numpy.zeros((col, 1)))

        iteration = 0
        shuffled_i = numpy.random.randint(raw)
        while True:
            iteration += 1
            # 求出单个样本残差
            difference = sample[shuffled_i, :] * w - label[shuffled_i, 0]
            w -= self.learning_rate * (sample[shuffled_i, :].reshape(-1, 1) * difference
                                       + 2 * self.penalty_factor * w) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach max_iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            shuffled_i = numpy.random.randint(raw)

        return w

    def mini_batch_gradient_descent(self, sample, label):
        """
        小批量梯度下降

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量

        Returns
        -------
        numpy.matrix 模型参数 w
        """
        print("* Start mini_batch_gradient_descent M_BGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        shuffled_i = numpy.random.permutation(raw)
        # 参数
        w = numpy.mat(numpy.zeros((col, 1)))
        # 小批量数量
        if raw < 100:
            return self.stochastic_gradient_descent(sample, label)
        elif raw < 500:
            mini_batch_size = 10
        else:
            mini_batch_size = 20

        iteration = 0
        while True:
            iteration += 1
            shuffled_sample = sample[shuffled_i]
            shuffled_label = label[shuffled_i]
            for i in range(0, raw - raw % mini_batch_size, mini_batch_size):
                temporary_sample = shuffled_sample[i: i + mini_batch_size, :]
                temporary_label = shuffled_label[i: i + mini_batch_size]
                w -= self.learning_rate * (temporary_sample.T * (temporary_sample * w - temporary_label)
                                           + 2 * self.penalty_factor * w) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration >= self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach max_iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            shuffled_i = numpy.random.permutation(raw)

        return w

    def mini_batch_gradient_descent_with_threshold(self, sample, label):
        """
        小批量梯度下降

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本，N行样本，p列属性
        label : numpy.matrix N * 1
            标签，列向量

        Returns
        -------
        numpy.matrix 模型参数 w
        """
        print("* Start mini_batch_gradient_descent M_BGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        shuffled_i = numpy.random.permutation(raw)
        # 参数
        w = numpy.mat(numpy.zeros((col, 1)))
        # 小批量数量
        if raw < 100:
            return self.stochastic_gradient_descent(sample, label)
        elif raw < 500:
            mini_batch_size = 10
        else:
            mini_batch_size = 20

        iteration = 0
        last_residuals = self.residuals(sample, label, w)
        while True:
            iteration += 1
            shuffled_sample = sample[shuffled_i]
            shuffled_label = label[shuffled_i]
            for i in range(0, raw - raw % mini_batch_size, mini_batch_size):
                temporary_sample = shuffled_sample[i: i + mini_batch_size, :]
                temporary_label = shuffled_label[i: i + mini_batch_size]
                w -= self.learning_rate * (temporary_sample.T * (temporary_sample * w - temporary_label)
                                           + 2 * self.penalty_factor * w) / raw

            residuals = self.residuals(sample, label, w)
            if self.debug_enable:
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if abs(residuals - last_residuals) < self.fit_threshold:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach fit_threshold')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            if iteration >= self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach max_iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + ' '.join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            shuffled_i = numpy.random.permutation(raw)
            last_residuals = self.residuals(sample, label, w)

        return w

    # 测试
    def predict(self, sample):
        sample = numpy.mat(sample)
        # 是否进行多项式转换
        if self.polynomial_degree != 0:
            sample = self.polynomial_features(sample, self.polynomial_degree)
        label = sample * self.w
        return label
