#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/30 9:04
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot
from sklearn.datasets import load_iris

import numpy
import leapy


if __name__ == "__main__":
    iris = load_iris()
    sample = numpy.array(iris.data[:100])[:, 1: 3]
    label = numpy.array(iris.target[:100]).T

    color_map = pyplot.get_cmap('viridis')

    pyplot.figure(0)
    pyplot.scatter(sample[:50, 0].T.tolist(), sample[:50, 1].T.tolist(), color=color_map(0.9), s=10, label="0")
    pyplot.scatter(sample[50:, 0].T.tolist(), sample[50:, 1].T.tolist(), color=color_map(0.6), s=10, label="1")

    model = leapy.HierarchicalClustering(debug_enable=True)
    model.fit(sample, 2)

    center = model.get_center()
    area = model.get_area()

    pyplot.scatter(center[:, 0].T.tolist(), center[:, 1].T.tolist(), color=color_map(0), s=10)

    pyplot.figure(1)
    # for i in range(len(area)):
    #     pyplot.scatter(area[i][:, 0].T, area[i][:, 1].T, s=10)
    pyplot.scatter(area[0][:, 0].T, area[0][:, 1].T, color=color_map(0.9), s=10)
    pyplot.scatter(area[1][:, 0].T, area[1][:, 1].T, color=color_map(0.6), s=10)
    pyplot.show()
