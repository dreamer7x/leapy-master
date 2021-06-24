#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/25 13:50
# @Author   : Mr. Fan

"""
高斯混合分布模型

这里重点论述了 EM算法
"""

from sklearn.cluster import KMeans

import math
import numpy


class GaussianMixture:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        # 维数
        self.p = 0
        # 分布数
        self.k = 0
        # 高斯模型
        self.gaussian_models = None
        # 优化算法
        self.algorithm = None
        # 迭代次数
        self.iteration = 0

    def fit(self, sample, k=1, algorithm=None, iteration=50):
        """
        拟合

        Parameters
        ----------
        sample : list / numpy.array / numpy.matrix N * p
            观测样本
        k : int
            高斯分布数
        algorithm : str
            算法
        iteration : int
            迭代次数上限
        """
        # 验证
        sample = numpy.mat(sample)
        raw, col = sample.shape

        # 初始化
        self.p = col
        self.k = k
        if algorithm is None:
            algorithm = "expectation_maximization"
        else:
            algorithm = algorithm
        self.algorithm = algorithm
        self.iteration = iteration

        # 拟合
        if self.algorithm == "expectation_maximization":
            self.gaussian_models = self.expectation_maximization(sample)
            return

    def expectation_step(self, sample, model):
        """
        E 步

        Parameters
        ----------
        sample : numpy.matrix
            样本
        model : list
            模型
        """
        hidden_sum = numpy.mat(numpy.zeros((sample.shape[0], 1), dtype=numpy.float64))

        for i in model:
            g = i["weight"] * self.get_gaussian(sample, i["average"], i["covariance"])
            i["hidden"] = g

            for j in range(sample.shape[0]):
                hidden_sum[j, 0] += i["hidden"][j, 0]

            i["hidden_sum"] = hidden_sum

        for i in model:
            i["hidden"] /= i["hidden_sum"]

    @staticmethod
    def maximization_step(sample, model):
        """
        M 步

        Parameters
        ----------
        sample : numpy.matrix
            样本
        model : list
            模型
        """
        for i in model:
            hidden = i["hidden"].T.tolist()[0]
            hidden_sum = sum(hidden)

            weight = hidden_sum / sample.shape[0]
            average = numpy.sum(hidden * sample, axis=0) / hidden_sum

            covariance = numpy.zeros((sample.shape[1], sample.shape[1]))
            for j in range(sample.shape[0]):
                difference = (sample[j, :] - average).reshape(-1, 1)
                covariance += hidden[j] * numpy.dot(difference, difference.T)
            covariance /= hidden_sum

            i["weight"] = weight
            i["average"] = average
            i["covariance"] = covariance

    def expectation_maximization(self, sample):
        """
        EM 优化算法

        Parameters
        ----------
        sample : numpy.matrix
            样本

        Returns
        -------
        list
            模型
        """
        # 初始化 这里使用 sklearn 提供的 KMeans 模块进行初始化
        k_means = KMeans().fit(sample)
        a = k_means.cluster_centers_
        models = []
        for i in range(self.k):
            model = {"weight": 1 / self.k,
                     "average": numpy.mat(a[i]),
                     "covariance": numpy.identity(self.p, dtype=numpy.float64),
                     "hidden": None,
                     "hidden_sum": None}
            models.append(model)
        # 迭代
        iteration = 0
        while True:
            iteration += 1
            self.expectation_step(sample, models)
            self.maximization_step(sample, models)
            if self.debug_enable:
                print("报告: GaussianMixed expectation_maximization 迭代完成")
                print("     log_likelihood: " + str(self.log_likelihood(models)))
            if iteration >= self.iteration:
                if self.debug_enable:
                    print("报告: GaussianMixed expectation_maximization 拟合完成")
                break
        return models

    # 获取初始协方差矩阵
    @staticmethod
    def get_covariance(sample, k):
        """
        初始化协方差矩阵

        Parameters
        ----------
        sample : numpy.matrix
            样本
        k : int
            高斯模型分布数

        Returns
        -------
        list [numpy.ndarray]
            协方差矩阵列表
        """
        covariances = []
        covariance = numpy.cov(sample.T)
        for i in range(k):
            # 初始的协方差矩阵源自于原始数据的协方差矩阵，且每个簇的初始协方差矩阵相同
            covariances.append(covariance)
        return covariances

    @staticmethod
    def log_likelihood(model):
        """
        对数似然函数

        Parameters
        ----------
        model : 模型对数似然函数负数和

        Returns
        -------
        float
            用以评估拟合程度
        """
        # 对数似然函数的负数来评判拟合程度
        error = numpy.log([i["hidden_sum"] for i in model])
        return -numpy.sum(error)

    # 计算概率密度
    @staticmethod
    def get_gaussian(sample, average, covariance):
        """
        多维高斯分布概率

        Parameters
        ----------
        sample : 样本
        average : numpy.ndarray N
            均值
        covariance : numpy.ndarray p * p
            协方差

        Returns
        -------

        """
        dimension = numpy.shape(covariance)[0]  # 维度
        # 加入单位矩阵 防止出现不可逆情况
        # 协方差矩阵的行列式
        covariance_determinant = numpy.linalg.det(covariance + numpy.eye(dimension) * 0.001)
        # 协方差矩阵的逆
        covariance_inverse = numpy.linalg.inv(covariance + numpy.eye(dimension) * 0.001)
        difference = sample - average
        # 概率密度公式
        gaussian = \
            numpy.diagonal(
                1.0 / numpy.power(2 * math.pi, dimension / 2) /
                numpy.power(numpy.abs(covariance_determinant), 1 / 2) * numpy.exp(
                    -1 / 2 * numpy.dot(numpy.dot(difference, covariance_inverse), difference.T))). \
            reshape(-1, 1)
        return gaussian
