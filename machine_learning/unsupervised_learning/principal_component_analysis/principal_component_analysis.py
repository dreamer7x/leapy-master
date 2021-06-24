#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/1 1:16
# @Author   : Mr. Fan

"""

"""

import numpy
import leapy


class PrincipalComponentAnalysis:

    def __init__(self, debug_enable=False):
        self.debug_enable = debug_enable

        self.covariance = None
        self.k = 0
        self.fit_threshold = 0
        self.e = None
        self.s = None

    @staticmethod
    def calculate_covariance(data):
        """

        Parameters
        ----------
        data : numpy.ndarray

        Returns
        -------

        """
        raw, col = data.shape

        return (data.T @ data) / raw

    @staticmethod
    def calculate_normalize(data):
        """
        Parameters
        ----------
        data

        Returns
        -------

        """
        data = data.copy()
        raw, col = data.shape

        for i in range(col):
            data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

        return data

    def generate(self, data, k=None, fit_threshold=0.8):
        """
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html

        Parameters
        ----------
        data
        k
        fit_threshold

        Returns
        -------

        """
        self.fit_threshold = fit_threshold

        # 1 normalize handwritten_numeral
        data = self.calculate_normalize(data)

        # 2 calculate covariance
        data_covariance = self.calculate_covariance(data)  # (n, n)

        # 3 do singular value decomposition
        # remember we feed covariance matrix in singular value decomposition,
        # since the covariance matrix is symmetry,
        # left singular vector and right singular vector is the same,
        # which means u is v, so we could use either one to do dim reduction
        model = leapy.SingularValueDecomposition()
        u, s, v = model.generate(data_covariance)  # U: principle components (n, n)

        # 4 calculate k
        if k is None:
            raw, col = s.shape
            self.k = raw
            total_contribution = numpy.sum(numpy.diagonal(s))
            contribution_rate = 0
            for i in range(raw):
                if self.debug_enable:
                    print("报告: PrincipalComponent generate 确定k值")
                    print("     contribution_rate: " + str(contribution_rate), end='')
                rate = s[i, i] / total_contribution
                contribution_rate += rate
                if self.debug_enable:
                    print(" + " + str(rate) + " = " + str(contribution_rate))
                if contribution_rate > self.fit_threshold:
                    self.k = i
                    break
        else:
            self.k = k

        self.s = s
        self.e = u

        return s, u

    def reduce_data(self, data):
        """
        压缩数据

        Parameters
        ----------
        data

        Returns
        -------

        """
        return data @ self.e[:, :self.k]

    def recover_data(self, data):
        """
        数据恢复

        Parameters
        ----------
        data

        Returns
        -------

        """
        raw, col = data.shape

        if col > self.e.shape[0]:
            raise ValueError('压缩数据维度大于特征向量数')

        return data @ self.e[:, :col].T
