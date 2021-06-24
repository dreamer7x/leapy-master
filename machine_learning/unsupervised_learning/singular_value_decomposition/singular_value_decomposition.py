#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/31 21:37
# @Author   : Mr. Fan

"""

"""

import numpy


class SingularValueDecomposition:

    def __init__(self, debug_enable=False):
        self.debug_enable = debug_enable

        self.shape = None
        self.u = None
        self.s = None
        self.v = None
        self.k = 0

    def generate(self, data, k=None):
        data = data
        raw, col = data.shape

        if raw > col:
            s, v = numpy.linalg.eigh(data.T @ data)
            # 进行降序排序
            i = numpy.argsort(s)[::-1]
            s = numpy.sort(s)[::-1]
            v = v[:, i]
            # 进行平方根处理
            s = numpy.diag(numpy.sqrt(s))
            s_inverse = numpy.linalg.inv(s)
            u = data @ v @ s_inverse

            u = numpy.pad(u, pad_width=((0, 0), (0, raw - col)))
            s = numpy.pad(s, pad_width=((0, raw - col), (0, 0)))
        else:
            s, u = numpy.linalg.eigh(data @ data.T)
            # 进行降序排序
            i = numpy.argsort(s)[::-1]
            s = numpy.sort(s)[::-1]
            u = u[:, i]
            # 进行平方根处理
            s = numpy.diag(numpy.sqrt(s))
            s_inverse = numpy.linalg.inv(s)
            v = s_inverse @ u @ data

            v = numpy.pad(v, pad_width=((0, col - raw), (0, 0)))
            s = numpy.pad(s, pad_width=((0, 0), (0, col - raw)))

        if k is None:
            self.k = u.shape[1]
        elif k < u.shape[1]:
            self.k = k
            u, s, v = u[:, :k], s[:k, :k], v[:k, :]
        else:
            self.k = u.shape[1]
        self.u, self.s, self.v = u, s, v

        return u, s, v

    def rebuild(self):
        raw, col = self.u.shape[0], self.v.shape[0]
        if raw > col:
            data = self.u @ self.s @ self.v.T
        else:
            data = self.u.T @ self.s @ self.v
        return data
