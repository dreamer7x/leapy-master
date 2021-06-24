#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 16:16
# @Author   : Mr. Fan

"""
回归树，解决回归预测问题，支持连续属性

Classes
-------
RegressionTree : 回归树模型
"""

import numpy
import pandas
import json


class RegressionTree:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 拟合阈值 对于信息增益应当大于 fit_threshold
        self.fit_threshold = 0
        # 惩罚项
        self.penalty_term = 0
        # 深度上限
        self.max_deep = 0
        # 根节点
        self.root = None
        # 离散化信息
        self.dispersed = None
        # 属性切割权重
        self.weights = None

    def fit(self, sample, label, max_deep=10, dispersed=None, fit_threshold=0.1, penalty_term=0, weights=None):
        """
        拟合

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本
        label : numpy.matrix N * 1
            标签
        max_deep : int
            最大深度
        dispersed : list
            离散化信息
        fit_threshold : float
            拟合阀值
        penalty_term : float
            惩罚系数
        weights : None, list
            属性切割权重
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label).T
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: IterativeDichotomiser3Tree fit 样本数，标签数不等")
            return

        # 如果离散化信息未提供 则设为默认离散
        if dispersed is None or dispersed is False:
            dispersed = [False] * sample.shape[1]
        if dispersed is True:
            dispersed = [True] * sample.shape[1]
        if col != len(dispersed):
            print("错误: Classification fit 属性数，离散化信息数不等")
            return

        if weights is None:
            weights = [1] * col
        if len(weights) != col:
            print("错误: Classification fit 属性数，属性切割权重数不等")
            return

        self.weights = weights
        self.dispersed = dispersed
        self.max_deep = max_deep
        self.fit_threshold = fit_threshold
        self.penalty_term = penalty_term

        self.root = self.regression_tree(sample, label)

    # 决策结点
    class TreeNode:
        def __init__(self, is_leaf=True, is_dispersed=False, label=None, dimension=None, value=None):
            # 是否为叶节点
            self.is_leaf = is_leaf
            # 是否离散化
            self.is_dispersed = is_dispersed
            # 分割维度
            self.dimension = dimension
            # 分割值
            self.value = value
            # 标签
            self.label = label
            # 左节点
            self.left = None
            # 右节点
            self.right = None

        # 预测
        def predict(self, sample):
            # 如果为叶节点
            if self.is_leaf is True:
                return self.label
            # 如果为离散值结点
            if self.is_dispersed:
                if sample[0, self.dimension] == self.value:
                    return self.left.forward(sample, )
                else:
                    return self.right.forward(sample, )
            else:
                if sample[0, self.dimension] <= self.value:
                    return self.left.forward(sample, )
                else:
                    return self.right.forward(sample, )

    @staticmethod
    def split_data(sample, label, dimension, value, is_dispersed=True):
        """
        切割数据

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本
        label : numpy.matrix N * 1
            标签
        dimension : int
            维度
        value : float
            切分点
        is_dispersed : bool
            是否为连续值

        Returns
        -------
        left_sample, left_label, right_sample, right_label numpy : numpy.matrix
            分割数据
        """
        sample = numpy.mat(sample)
        label = numpy.mat(label)
        raw, col = sample.shape
        left_sample, left_label = [], []
        right_sample, right_label = [], []
        for i in range(raw):
            if is_dispersed:
                if sample[i, dimension] == value:
                    left_sample.append(sample[i, :].tolist()[0])
                    left_label.append(label[i, :].tolist()[0])
                else:
                    right_sample.append(sample[i, :].tolist()[0])
                    right_label.append(label[i, :].tolist()[0])
            else:
                if sample[i, dimension] <= value:
                    left_sample.append(sample[i, :].tolist()[0])
                    left_label.append(label[i, :].tolist()[0])
                else:
                    right_sample.append(sample[i, :].tolist()[0])
                    right_label.append(label[i, :].tolist()[0])
        left_sample = numpy.mat(left_sample)
        left_label = numpy.mat(left_label)
        right_sample = numpy.mat(right_sample)
        right_label = numpy.mat(right_label)
        return left_sample, left_label, right_sample, right_label

    def least_squares(self, sample, label, is_dispersed=False, axis=0):
        raw, col = sample.shape
        label_split = {}
        if self.debug_enable:
            print("报告: RegressionTree least_squares 计算最小二乘值")
            print("     raw: " + str(raw))
            print("     axis: " + str(axis))
        for i in range(raw):
            if sample[i, axis] not in label_split:
                label_split[sample[i, axis]] = [[], []]
                for j in range(raw):
                    if is_dispersed:
                        if sample[j, axis] == sample[i, axis]:
                            label_split[sample[i, axis]][0].append(label[j, 0])
                        else:
                            label_split[sample[i, axis]][1].append(label[j, 0])
                    else:
                        if sample[j, axis] <= sample[i, axis]:
                            label_split[sample[i, axis]][0].append(label[j, 0])
                        else:
                            label_split[sample[i, axis]][1].append(label[j, 0])
        # 遍历计算最小二乘数
        squares = []
        for i in label_split.keys():
            if len(label_split[i][0]) == 0 or len(label_split[i][1]) == 0:
                squares.append(float("inf"))
                continue
            left_mean = sum(label_split[i][0]) / len(label_split[i][0])
            right_mean = sum(label_split[i][1]) / len(label_split[i][1])
            left_variance = sum([(i0 - left_mean) ** 2 for i0 in label_split[i][0]])
            right_variance = sum([(i1 - right_mean) ** 2 for i1 in label_split[i][1]])
            squares.append(left_variance + right_variance)
        min_square = min(enumerate(squares), key=lambda s: s[1])
        value = list(label_split.keys())[min_square[0]]
        return value, min_square[1]

    def gain_dimension(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        least_squares = []
        for i in range(col):
            if self.dispersed[i]:
                value = self.least_squares(sample, label, is_dispersed=True, axis=i)
                least_squares.append([i, value[0], value[1] / self.weights[i]])
            else:
                value = self.least_squares(sample, label, is_dispersed=False, axis=i)
                least_squares.append((i, value[0], value[1] / self.weights[i]))
        gain = sorted(least_squares, key=lambda f: f[-1])[0]
        return gain

    def regression_tree(self, sample, label, deep=1):
        """
        构造回归树

        Parameters
        ----------
        sample : numpy.matrix
            样本
        label : numpy.matrix
            标签
        deep : int
            深度
        """
        raw, col = sample.shape
        label_count = pandas.value_counts(label.T.tolist()[0])
        # 1 如果只有只用类别，则返回叶结点
        if label_count.shape[0] == 1:
            return self.TreeNode(is_leaf=True,
                                 label=sum(label.T.tolist()[0]) / raw)
        # 2 如果特性为空，则返回叶节点
        if col == 0:
            return self.TreeNode(is_leaf=True,
                                 label=sum(label.T.tolist()[0]) / raw)

        # - 如果达到深度上限，则返回叶节点
        if deep >= self.max_deep:
            if self.debug_enable:
                print("报告: RegressionTree regression_tree 达到深度上限")
            return self.TreeNode(is_leaf=True,
                                 label=sum(label.T.tolist()[0]) / raw)

        # 3 获取最佳 切割维度 和 切割值
        dimension, value, least_square = self.gain_dimension(sample, label)
        if self.debug_enable:
            print("报告: RegressionTree regression_tree 获取最佳切割位置")
            print("     dimension: " + str(dimension))
            print("     value: " + str(value))
            print("     least_square: " + str(least_square))

        # 4 如果基尼指数小于拟合阈值
        if least_square < self.fit_threshold:
            if self.debug_enable:
                print("报告: RegressionTree regression_tree 拟合达到理想状态")
            return self.TreeNode(is_leaf=True,
                                 label=sum(label.T.tolist()[0]) / raw)

        # 删除属性列
        # temporary_sample = numpy.delete(sample, dimension, axis=1)
        temporary_sample = sample

        # 5 初始化结点
        node = self.TreeNode(is_leaf=False,
                             is_dispersed=self.dispersed[dimension],
                             dimension=dimension,
                             value=value)

        left_sample, left_label, right_sample, right_label = \
            self.split_data(temporary_sample, label, dimension, value, self.dispersed[dimension])

        # 6 递归生成左右结点
        node.left = self.regression_tree(left_sample, left_label)
        node.right = self.regression_tree(right_sample, right_label)

        return node

    def get_leaf(self, node):
        """
        获取叶结点标记集合

        Parameters
        ----------
        node : TreeNode
            回归树结点

        Returns
        -------
        labels : list
            标签列表
        """
        if node.is_leaf:
            return node.label_set
        labels = []
        for i in node.nodes.keys():
            labels.append(self.get_leaf(node[i]))
        return labels

    def prune_classification_and_regression_tree(self, node, sample, label):
        raw, col = sample.shape
        # 如果验证样本为空 进行塌陷处理
        if raw == 0:
            labels = self.get_leaf(node)
            return self.TreeNode(is_leaf=True,
                                 label=sum(labels) / len(list(labels)))
        # 如果为叶节点
        if not node.is_leaf:
            left_sample, left_label, right_sample, right_label = \
                self.split_data(sample, label, node.dimension, node.value)
            left_node = self.prune_classification_and_regression_tree(node.left, left_sample, left_label)
            right_node = self.prune_classification_and_regression_tree(node.right, right_sample, right_label)
            if left_node.is_leaf and right_node.is_leaf:
                error_leaf = \
                    (sum([(i - left_node.label) ** 2 for i in left_label.T.tolist()]) +
                     sum([(i - right_node.label) ** 2 for i in right_label.T.tolist()])) / \
                    (left_label.shape[0] + right_node.shape[0])
                mean = (left_node.label + right_node.label) / 2
                error_tree = \
                    sum([(i - mean) ** 2 for i in (left_label.T.tolist() + right_label.T.tolist())]) / \
                    (left_label.shape[0] + right_node.shape[0])
                # 如果满足剪枝条件
                if error_tree < error_leaf + self.penalty_term:
                    return self.TreeNode(is_leaf=True,
                                         label=mean)
                else:
                    return node
            else:
                return node
        return node

    def test(self, sample, label):
        sample = numpy.mat(sample)
        label = numpy.mat(label)
        self.root = self.prune_classification_and_regression_tree(self.root, sample, label)

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            label[i, 0] = self.root.forward(sample[i, :], )
        return label

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: Classification fit 样本数，标签数不等")
            return

        label_predict = self.predict(sample)
        score = 0
        for i in range(raw):
            score += (label_predict[i, 0] - label[i, 0]) ** 2
            print(label_predict[i, 0], label[i, 0])
        return score

    def load(self, path):
        file = open(path, "r")
        self.decode(eval(file.read()))

    def save(self, path):
        file = open(path, "w")
        file.write(str(self.encode()))

    def decode(self, encode):
        def d(e):
            node = self.TreeNode(
                is_leaf=e["is_leaf"],
                is_dispersed=e["is_dispersed"],
                dimension=e["dimension"],
                value=e["value"],
                label=e["label"]
            )
            if not node.is_leaf:
                node.left = d(e["left"])
                node.right = d(e["right"])
            return node
        self.root = d(encode)

    def encode(self):
        def e(node):
            if node is None:
                return None
            encode = {
                "is_leaf": node.is_leaf,
                "is_dispersed": node.is_dispersed,
                "dimension": node.dimension,
                "value": node.value,
                "label": node.label,
                "left": e(node.left),
                "right": e(node.right)
            }
            return encode
        return e(self.root)
