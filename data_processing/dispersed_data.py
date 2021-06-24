#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/9 19:05
# @Author   : Mr. Fan

"""

"""

import numpy


def to_categorical(sample, col=None):
    """
    非连续数据数值化，进制化

    Parameters
    ----------
    sample : numpy.ndarray
        样本
    col :
        属性数

    Returns
    -------
    """
    if not col:
        col = numpy.amax(sample) + 1
    one_hot = numpy.zeros((sample.shape[0], col))
    one_hot[numpy.arange(sample.shape[0]), sample] = 1
    return one_hot
