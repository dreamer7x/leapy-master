#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 23:42
# @Author   : Mr. Fan

"""
标准线性回归

Classes
-------
StandardLinearRegression : 标准线性回归类对象
"""

import numpy
import itertools
import datetime


class LassoLinearRegression:

    def __init__(self, debug_enable=False, threshold_enable=True, fit_threshold=0.1,
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
                    \begin{aligned} Update \ w_p: \\
                    m_p & = & x_{ip}(y_i - \sum_{i \not= p} x_{ij} w_{j}) \\
                    w_p & = & \left\{ \begin{aligned} \frac{m_p + \lambda / 2}{x_i^T x_i}, m_p < -\frac{\lambda}{2} \\
                    0, -\frac{\lambda}{2} \leq m_p \leq \frac{\lambda}{2} \\
                    \frac{m_p - \lambda / 2}{x_i^T x_i}, m_p < -\frac{\lambda}{2} \end{aligned} \right. \end{aligned}
            2. batch_gradient_descent or BGD
                math:
                    w = w - \alpha \frac{x_i(x_i w - y_i) + \lambda sign(w)}{N}
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

        learning_rate : float 学习率
        polynomial_degree : int 多项式深度
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
        # 优化算法
        if algorithm is None:
            self.algorithm = "mini_batch_gradient_descent"
        else:
            self.algorithm = algorithm
        # 惩罚系数
        self.penalty_factor = penalty_factor
        # 迭代次数
        self.max_iteration = max_iteration
        # 多项式深度
        self.polynomial_degree = polynomial_degree

    def fit(self, train_sample, train_label):
        """
        Parameters
        ----------
        train_sample : numpy.array      shape of N * p
                       numpy.matrix     shape of N * p
                       list             shape of N * [p]
        train_label : numpy.array       shape of N * 1 / 1 * N
                      numpy.matrix      shape of N * 1 / 1 * N
                      list              shape of N
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
            if self.threshold_enable:
                self.w = self.math_solving_with_threshold(train_sample, train_label)
            else:
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
                print("* Warning from StandardLinearRegression.fit() ")
                print("  Optimization algorithm not specified.       ")
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
        residuals = label - sample * w
        residuals = residuals.T * residuals
        return residuals[0, 0]

    # 多项式特征转换
    @staticmethod
    def polynomial_features(sample, degree):
        """
        Parameters
        ----------
        sample : numpy.matrix shape of N * p_0
            样本，N行样本，p_0列属性
        degree : int
            转换深度

        Returns
        -------
        numpy.matrix N * p_1
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
            # Copy matrix by using tuple index:
            #   matrix[:, (0, 0, ...)] shape of N * p -> matrix shape of N * (p *len(j))
            new_sample[:, i] = numpy.prod(sample[:, j], axis=1)

        return new_sample

    def math_solving(self, sample, label):
        print('* Start math_solving')
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        # 样本数，属性数
        raw, col = sample.shape
        # 参数
        w = numpy.mat(numpy.zeros((col, 1)))

        iteration = 0
        while True:
            iteration += 1

            for p in range(col):
                n = numpy.zeros(col)
                m_0 = numpy.mat(numpy.zeros((col, 1)))
                m_1 = numpy.mat(numpy.zeros((col, 1)))
                temporary_sample = sample[:, p].reshape(-1, 1)
                n[p] = (temporary_sample.T * temporary_sample)[0, 0]

                for i in range(raw):
                    for j in range(col):
                        if p != j:
                            m_1[j, 0] = sample[i, j] * w[j, 0]
                    m_0[p, 0] += sample[i, p] * (label[i, 0] - numpy.sum(m_1))

                if m_0[p] < -self.penalty_factor / 2:
                    w[p, 0] = (m_0[p] + self.penalty_factor / 2) / n[p]
                elif m_0[p] > self.penalty_factor / 2:
                    w[p, 0] = (m_0[p] - self.penalty_factor / 2) / n[p]
                else:
                    w[p, 0] = 0

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration >= self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

        return w

    def math_solving_with_threshold(self, sample, label):
        print('* Start math_solving')
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

            for p in range(col):
                n = numpy.zeros(col)
                m_0 = numpy.mat(numpy.zeros((col, 1)))
                m_1 = numpy.mat(numpy.zeros((col, 1)))
                temporary_sample = sample[:, p]
                n[p] = (temporary_sample.T * temporary_sample)[0, 0]

                for i in range(raw):
                    for j in range(col):
                        if p != j:
                            m_1[j, 0] = sample[i, j] * w[j, 0]
                    m_0[p, 0] += sample[i, p] * (label[i, 0] - numpy.sum(m_1))

                if m_0[p] < -self.penalty_factor / 2:
                    w[p, 0] = (m_0[p] + self.penalty_factor / 2) / n[p]
                elif m_0[p] > self.penalty_factor / 2:
                    w[p, 0] = (m_0[p] - self.penalty_factor / 2) / n[p]
                else:
                    w[p, 0] = 0

            residuals = self.residuals(sample, label, w)
            if self.debug_enable:
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if abs(last_residuals - residuals) < self.fit_threshold:
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
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            last_residuals = residuals

        return w

    def batch_gradient_descent(self, sample, label):
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
                                           + self.penalty_factor * numpy.sign(w)) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

        return w

    def batch_gradient_descent_with_threshold(self, sample, label):
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
                w -= self.learning_rate * (sample[i, :].reshape(-1, 1) * difference +
                                           self.penalty_factor * numpy.sign(w)) / raw

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

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            last_residuals = residuals

        return w

    def stochastic_gradient_descent(self, sample, label):
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
            difference = sample[shuffled_i, :] * w - label[shuffled_i]
            w -= self.learning_rate * (sample[shuffled_i, :].reshape(-1, 1) * difference
                                       + self.penalty_factor * numpy.sign(w)) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            shuffled_i = numpy.random.randint(raw)

        return w

    def mini_batch_gradient_descent(self, sample, label):
        print("* Start mini_batch_stochastic_gradient_descent M_BGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        raw, col = sample.shape
        w = numpy.mat(numpy.zeros((col, 1)))
        if raw < 100:
            return self.stochastic_gradient_descent(sample, label)
        elif raw < 500:
            mini_batch_size = 10
        else:
            mini_batch_size = 20

        iteration = 0
        shuffled_i = numpy.random.permutation(raw)
        while True:
            iteration += 1
            shuffled_sample = sample[shuffled_i]
            shuffled_label = label[shuffled_i]
            for i in range(0, raw - raw % mini_batch_size, mini_batch_size):
                temporary_sample = shuffled_sample[i: i + mini_batch_size, :]
                temporary_label = shuffled_label[i: i + mini_batch_size]
                w -= self.learning_rate * (temporary_sample.T.dot(temporary_sample.dot(w) - temporary_label)
                                           + self.penalty_factor * numpy.sign(w)) / raw

            if self.debug_enable:
                residuals = self.residuals(sample, label, w)
                print('   ', '{0:03d}'.format(iteration), ':', residuals)

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]))
                break

            shuffled_i = numpy.random.permutation(raw)
        return w

    def mini_batch_gradient_descent_with_threshold(self, sample, label):
        print("* Start mini_batch_stochastic_gradient_descent M_BGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : residuals                       ')
            print('------------------------------------------------')

        # 样本数，属性数
        raw, col = sample.shape
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
        shuffled_i = numpy.random.permutation(raw)
        last_residuals = self.residuals(sample, label, w)
        while True:
            iteration += 1
            shuffled_sample = sample[shuffled_i]
            shuffled_label = label[shuffled_i]
            for i in range(0, raw - raw % mini_batch_size, mini_batch_size):
                temporary_sample = shuffled_sample[i: i + mini_batch_size, :]
                temporary_label = shuffled_label[i: i + mini_batch_size]
                w -= self.learning_rate * (temporary_sample.T.dot(temporary_sample.dot(w) - temporary_label)
                                           + self.penalty_factor * numpy.sign(w)) / raw

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

            if iteration > self.max_iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n    ' + " ".join([str(i) for i in w[:, 0].tolist()]))
                break

            shuffled_i = numpy.random.permutation(raw)
            last_residuals = residuals

        return w

    def predict(self, sample):
        sample = numpy.mat(sample)
        # 是否进行多项式转换
        if self.polynomial_degree != 0:
            sample = self.polynomial_features(sample, self.polynomial_degree)
        label = sample * self.w
        return label
