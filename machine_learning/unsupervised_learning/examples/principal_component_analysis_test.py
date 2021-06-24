#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/1 1:49
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot
from scipy import io

import seaborn
import leapy
import numpy


def show_images(d, n):
    """
    展示前 n 张图片
    """
    picture_size = int(numpy.sqrt(d.shape[1]))
    grid_size = int(numpy.sqrt(n))

    images = d[:n, :]

    figure, array = pyplot.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            array[r, c].imshow(images[grid_size * r + c].reshape((picture_size, picture_size)))
            pyplot.xticks(numpy.array([]))
            pyplot.yticks(numpy.array([]))


if __name__ == "__main__":
    seaborn.set(context="notebook", style="white")

    data = leapy.load_faces()
    print(data.shape)

    show_images(data, n=64)
    pyplot.show()

    model = leapy.PrincipalComponentAnalysis(debug_enable=True)
    _, e = model.generate(data, fit_threshold=1)
    print(e.shape)
    print(model.k)

    reduce_data = model.reduce_data(data)
    show_images(reduce_data, n=64)
    pyplot.show()

    recover_data = model.recover_data(reduce_data)
    show_images(recover_data, n=64)
    pyplot.show()
