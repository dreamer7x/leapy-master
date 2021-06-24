#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/26 16:59
# @Author   : Mr. Fan

from sklearn.datasets import load_iris

import numpy
import leapy


if __name__ == "__main__":
    iris = load_iris()
    sample = iris.data

    model = leapy.GaussianMixture(debug_enable=True)
    model.fit(sample, k=3)
