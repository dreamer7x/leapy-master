#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/24 13:26
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import leapy
import numpy
import pandas

if __name__ == "__main__":
    # 取实验数据 为字典类型
    # {"handwritten_numeral":[[...], [...], ...], "target":[...]}
    iris = load_iris()
    # 将其注入DataFrame类型
    # 参数1: 样本数据(矩阵形式)
    # 参数2: 属性名(属性名字符串列表)
    data_frame = pandas.DataFrame(iris.data, columns=iris.feature_names)
    data_frame['label'] = iris.target
    # 覆写属性名 覆写数量不匹配将报错
    data_frame.columns = ['x1', 'x2', 'x3', 'x4', 'label']

    # 读取
    data = numpy.array(data_frame.iloc[:100, [0, 1, -1]])
    sample, label = data[:, :-1], data[:, -1]
    sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.3)

    model = leapy.LogisticRegressionClassifier(max_iteration=1000)
    model.fit(sample_train, label_train)

    line_x = numpy.arange(4, 8)
    line_y = -(model.w[0, 0] * line_x + model.b) / model.w[1, 0]
    pyplot.plot(line_x, line_y)

    color_map = pyplot.get_cmap('viridis')
    pyplot.scatter(sample[:50, 0].T.tolist(), sample[:50, 1].T.tolist(),
                   label='0', color=color_map(0.9), s=10)
    pyplot.scatter(sample[50:, 0].T.tolist(), sample[50:, 1].T.tolist(),
                   label='1', color=color_map(0.5), s=10)
    pyplot.legend()
    pyplot.show()

    print(model.score(sample_test, label_test))
