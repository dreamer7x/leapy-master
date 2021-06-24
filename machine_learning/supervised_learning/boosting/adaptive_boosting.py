#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/25 0:34
# @Author   : Mr. Fan

"""
adaptive_boosting
-----------------
二分类算法

模型为加法模型，损失函数为指数函数，学习算法为前向分布算法
"""

import numpy


class AdaptiveBoosting:
    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        # 弱选择器数量上限
        self.max_estimator_number = 0
        # 学习率
        self.learning_rate = 0
        # 弱分类器集合
        self.estimator = None
        # 训练样本
        self.sample = None
        # 训练标签
        self.label = None
        # 初始化变量权重
        self.w = None
        # 弱分类器权重
        self.a = None

    def fit(self, sample, label, estimator_number=50, learning_rate=1.0):
        # 初始化系数
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(label)

        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: Classification fit 样本数，标签数不等")
            return

        self.sample = sample
        self.label = numpy.mat(label)

        self.max_estimator_number = estimator_number
        self.learning_rate = learning_rate

        self.estimator = []
        self.w = numpy.mat([[1 / raw] * raw]).T
        self.a = numpy.mat(numpy.zeros((self.max_estimator_number, 1)))

        while True:
            split_e = float("inf")
            split_value = None
            split_label = None
            split_direction = None
            split_axis = None

            for j in range(col):
                value, direction, e, label = self.get_g(j)

                if e < split_e:
                    split_e = e
                    split_value = value
                    split_label = label
                    split_direction = direction
                    split_axis = j

                # 已达到理想拟合状态
                if split_e == 0:
                    break

            self.a[len(self.estimator), 0] = self.get_a(split_e)
            self.update_w(split_label, self.a[len(self.estimator), 0])

            self.estimator.append((split_axis, split_value, split_direction))

            if self.debug_enable:
                print("报告: AdaptiveBoosting fit 迭代")
                print("     number: " + str(len(self.estimator)))
                print("     split_axis: " + str(split_axis))
                print("     split_value: " + str(split_value))
                print("     split_e: " + str(split_e))

            if len(self.estimator) >= self.max_estimator_number:
                if self.debug_enable:
                    print("报告: AdaptiveBoosting fit 拟合完成 达到弱选择器数量上限")
                break

    # 构造弱选择器模型
    def get_g(self, j):
        raw, col = self.sample.shape
        e = float("inf")
        split_value = 0
        direction = None
        label = None
        feature = self.sample[:, j].T.tolist()[0]
        min_value = min(feature)
        max_value = max(feature)
        step = (max_value - min_value + self.learning_rate) // self.learning_rate
        for j in range(1, int(step)):
            value = min_value + self.learning_rate * j
            # 如果值并不属于属性值列表
            if value not in feature:
                positive_label = numpy.mat([[1 if feature[i] > value else -1 for i in range(raw)]]).T
                positive_w = sum([self.w[i, 0] for i in range(raw) if positive_label[i, 0] != self.label[i, 0]])

                negative_label = numpy.mat([[-1 if feature[i] > value else 1 for i in range(raw)]]).T
                negative_w = sum([self.w[i, 0] for i in range(raw) if negative_label[i, 0] != self.label[i, 0]])

                if positive_w < negative_w:
                    e_weight = positive_w
                    temporary_label = positive_label
                    direction = 'positive'
                else:
                    e_weight = negative_w
                    temporary_label = negative_label
                    direction = 'negative'

                if e_weight < e:
                    e = e_weight
                    label = temporary_label
                    split_value = value
        return split_value, direction, e, label

    # 权值更新
    def update_w(self, label, a):
        for i in range(label.shape[0]):
            self.w[i, 0] = self.w[i, 0] * numpy.exp(-1 * a * self.label[i, 0] * label[i, 0]) / self.get_z(label, a)

    # 规范化因子
    def get_z(self, label, a):
        return sum([self.w[i, 0] *
                    numpy.exp(-1 * a * self.label[i, 0] * label[i, 0])
                    for i in range(self.sample.shape[0])])

    @staticmethod
    def get_a(e):
        return 0.5 * numpy.log((1 - e) / e)

    @staticmethod
    def g(value, split_value, direction):
        if direction == "positive":
            return 1 if value > split_value else -1
        else:
            return -1 if value > split_value else 1

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(numpy.zeros((raw, 1)))
        for i in range(raw):
            for j in range(len(self.estimator)):
                axis, split_value, direction = self.estimator[j]
                value = sample[i, axis]
                label[i, 0] += self.a[j, 0] * self.g(value, split_value, direction)
            label[i, 0] = 1 if label[i, 0] > 0 else -1
        return label

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: AdaptiveBoosting score 样本数，标签数不等")
            return
        right_count = 0
        label_predict = self.predict(sample)
        for i in range(raw):
            if label_predict[i, 0] == label[i, 0]:
                right_count += 1
        score = right_count / raw
        if self.debug_enable:
            print("报告: AdaptiveBoosting score 评分")
            print("     score: " + str(score))
        return score
