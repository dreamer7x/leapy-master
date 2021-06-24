#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 23:42
# @Author   : Mr. Fan

"""
局部加权线性回归

Classes
-------
LocalLinearRegression : 局部加权线性回归类对象
"""

import numpy


class LocalLinearRegression:

    def __init__(self, debug_enable=False, fit_factor=1.0):
        """
        Parameters
        ----------
        debug_enable : bool
        fit_factor : float
        """
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        self.train_sample = None
        self.train_label = None
        self.fit_factor = fit_factor

    # 以array，matrix或list的形式传入样本和标签作为参数
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

        self.train_sample = train_sample
        self.train_label = train_label

    def predict(self, sample):
        """
        预测

        Parameters
        ----------
        sample : numpy.ndarray shape of N * p

        Returns
        -------
        numpy.matrix N * 1
        预测标签
        """
        sample = numpy.array(sample)
        raw, col = sample.shape
        label = numpy.zeros((raw, 1))

        for i in range(raw):
            # 生成权重矩阵
            # 与标准线性回归不同，要对每一个预测样本生成权重矩阵
            weights = numpy.eye(self.train_sample.shape[0])
            for j in range(self.train_sample.shape[0]):
                # 计算权重
                #
                # 1.从几何的角度进行解释: e ^ (测试点与样本点的距离 / -2 * 权重系数 ** 2)
                # 由上式可知，权重系数越小，指数越容易往负方向偏移，对测试点与样本点的距离越敏感，拟合度越高（拟合度过高越容易过拟合，谨慎选取权重系数）
                #
                # 2.由于 difference * difference.T / -2.0 * ratio ** 2 < 0 所以权重值始终在 [0, 1] 之间
                difference = sample[i, :] - self.train_sample[j, :]
                weights[j, j] = numpy.exp(difference * difference.T / (-2.0 * self.fit_factor ** 2))  # 计算权重
            w = self.train_sample.T * weights * self.train_sample
            # 如果 w 可逆
            if numpy.linalg.det(w) == 0.0:
                print('------------------------------------------------')
                print('* Error from LocalLinearRegression.predict()    ')
                print('  The solution matrix is not invertible         ')
                print('------------------------------------------------')
                return numpy.zeros(1)
            # 局部加权线性回归求解公式
            w = numpy.linalg.inv(w) * self.train_sample.T * weights * self.train_label
            label[i, 0] = sample[i, :] * w

        return label
