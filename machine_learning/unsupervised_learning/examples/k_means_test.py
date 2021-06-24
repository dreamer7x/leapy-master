#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/30 19:56
# @Author   : Mr. Fan

"""

"""


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from matplotlib import pyplot
    from itertools import cycle

    import numpy
    import leapy

    data, label = make_blobs(centers=4, cluster_std=0.5)

    model = leapy.KMeans(debug_enable=True)
    model.fit(data, k=4)
    clusters = model.clusters
    centers = numpy.array(model.centers)

    color = 'bgrym'
    for i, j in zip(cycle(color), clusters.values()):
        partial_data = data[j]
        pyplot.scatter(partial_data[:, 0], partial_data[:, 1], color=i)
    pyplot.scatter(centers[:, 0], centers[:, 1], color='k', marker='*', s=100)
    pyplot.show()
