#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 16:16
# @Author   : Mr. Fan

"""
ID3决策树，最基本的决策树算法，解决多分类问题，仅支持离散属性

Classes
-------
IterativeDichotomiser3Tree : ID3 决策树模型
"""

import numpy
import pandas
import math


class IterativeDichotomiser3Tree:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 拟合阈值 对于信息增益应当大于 fit_threshold
        self.fit_threshold = 0
        # 深度上限
        self.max_deep = 0
        # 根节点
        self.root = None
        # 校正 是否使用C4.5的方法对经验熵进行校正
        self.adjust_enable = False

    def fit(self, sample, label, fit_threshold=0.1, max_deep=10, adjust_enable=False):
        """
        拟合

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签
        max_deep : int
            最大深度
        fit_threshold : float
            拟合阀值
        adjust_enable : bool
            校正
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: IterativeDichotomiser3Tree fit 样本数，标签数不等")
            return

        self.max_deep = max_deep
        self.fit_threshold = fit_threshold
        self.adjust_enable = adjust_enable

        self.root = self.iterative_dichotomiser_3(sample, label)

    # 决策结点
    class TreeNode:
        def __init__(self, is_leaf=True, label=None, dimension=None):
            # 是否为叶节点
            self.is_leaf = is_leaf
            # 分割维度
            self.dimension = dimension
            # 标签
            self.label = label
            # 子节点
            self.nodes = {}

        # 添加节点
        def add_node(self, value, node):
            self.nodes[value] = node

        # 预测
        def predict(self, sample):
            """
            预测

            Parameters
            ----------
            sample : matrix N * 1
                单样本

            Returns
            -------
            label : ...
                标签
            """
            # 如果为叶节点
            if self.is_leaf is True:
                return self.label
            else:
                key = sample[0, self.dimension]
                sample = numpy.delete(sample, self.dimension, axis=1)
                try:
                    return self.nodes[key].forward(sample, )
                except KeyError:
                    return self.label

    @staticmethod
    def empirical_entropy(label):
        """
        计算经验熵

        Parameters
        ----------
        label : list
            标签列表
        Returns
        -------
        entropy : float
            经验熵
        """
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        label_count = pandas.value_counts(label.T.tolist()[0])
        sample_size = sum(label_count.values)
        entropy = -sum([(i / sample_size) * math.log(i / sample_size, 2) for i in label_count.values])
        return entropy

    def conditional_empirical_entropy(self, sample, label, axis=0):
        """
        计算经验条件熵

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签
        axis : int
            属性列位置

        Returns
        -------
        entropy : float
            经验条件熵
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape
        sample_split = {}
        for i in range(raw):
            if sample[i, axis] not in sample_split:
                sample_split[sample[i, axis]] = []
            sample_split[sample[i, axis]].append(label[i, 0])
        entropy = sum([(len(i) / raw) * self.empirical_entropy(i) for i in sample_split.values()])
        return entropy

    def gain_dimension(self, sample, label):
        """
        计算最佳特性

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签

        Returns
        -------
        dimension : int
            分类维度
        information_gain : float
            信息增益
        """
        sample = numpy.mat(sample)
        label = numpy.mat(label)
        raw, col = sample.shape
        empirical_entropy = self.empirical_entropy(label.T)
        entropy = []
        for i in range(col):
            # 计算信息增益
            information_gain_ratio = empirical_entropy - self.conditional_empirical_entropy(sample, label, axis=i)
            if self.adjust_enable:
                # C4.5，计算信息增益比，这里是与ID3不同的地方
                information_gain_ratio = information_gain_ratio / self.empirical_entropy(sample[:, i])
            entropy.append((i, information_gain_ratio))
        gain = max(entropy, key=lambda f: f[-1])
        return gain

    def iterative_dichotomiser_3(self, sample, label, deep=1):
        """
        构造ID3分类树

        Parameters
        ----------
        sample : numpy.matrix
            样本
        label : numpy.matrix
            标签
        deep : int
            深度

        Returns
        -------
        node : TreeNode
            ID3 树结点
        """
        raw, col = sample.shape
        # ID3 算法
        # 1 若标签同类，则返回单节点树
        label_count = pandas.value_counts(label.T.tolist()[0])
        if label_count.shape[0] == 1:
            node = self.TreeNode(is_leaf=True,
                                 label=label[0, 0])
            return node

        # 2 若特征为空，则返回单节点树，将占比最大标签作为结点标签
        if col == 0:
            node = self.TreeNode(is_leaf=True,
                                 label=max(label_count.items(), key=lambda i: i[-1])[0])
            return node

        # - 若达到深度上限
        if deep >= self.max_deep:
            node = self.TreeNode(is_leaf=True,
                                 label=max(label_count.items(), key=lambda i: i[-1])[0])
            return node

        # 3 获取信息增益最大特征
        dimension, information_gain = self.gain_dimension(sample, label)
        if self.debug_enable:
            print("报告: IterativeDichotomiser3Tree iterative_dichotomiser_3_tree 获取信息增益最大特征")
            print("     dimension: " + str(dimension))
            print("     information_gain: " + str(information_gain))

        # 4 信息增益小于阈值,则置为单节点树，将占比最大标签作为结点标签
        if information_gain < self.fit_threshold:
            if self.debug_enable:
                print("报告: IterativeDichotomiser3Tree iterative_dichotomiser_3_tree 拟合达到理想状态")
            node = self.TreeNode(is_leaf=True,
                                 label=max(label_count.items(), key=lambda i: i[-1])[0])
            return node

        # 5 初始化结点
        node = self.TreeNode(is_leaf=False,
                             dimension=dimension,
                             label=max(label_count.items(), key=lambda i: i[-1])[0])

        values = pandas.value_counts(sample[:, dimension].T.tolist()[0]).keys()

        for value in values:
            split = [[i, j] for i, j in zip(sample.tolist(), label.tolist()) if i[dimension] == value]
            # - 删除分类属性列
            sample_split = numpy.mat(numpy.delete(numpy.mat([i[0] for i in split]), dimension, axis=1))
            label_split = numpy.mat([i[-1] for i in split])
            # 6 递归生成树
            # 对于非单节点 则递归生成所有子节点
            add_node = self.iterative_dichotomiser_3(sample_split, label_split, deep=deep + 1)
            node.add_node(value, add_node)

        return node

    # 获取叶结点标记集合
    def get_leaf(self, node):
        """
        获取叶节点集合
        Parameters
        ----------
        node

        Returns
        -------

        """
        if node.is_leaf:
            return node.label_set
        labels = []
        for i in node.nodes.keys():
            labels.append(self.get_leaf(node[i]))
        return labels

    def predict(self, sample):
        """
        预测

        Parameters
        ----------
        sample : matrix N * p
            样本

        Returns
        -------
        label_predict : matrix N * 1
            标签
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label_predict = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            label_predict[i, 0] = self.root.forward(sample[i, :], )
        return label_predict

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: IterativeDichotomiser3Tree score 样本数，标签数不等")
            return

        label_predict = self.predict(sample)
        count = 0
        for i in range(raw):
            print(label_predict[i, 0], label[i, 0])
            if label_predict[i, 0] == label[i, 0]:
                count += 1
        score = count / raw
        if self.debug_enable:
            print("报告: IterativeDichotomiser3Tree score 计算正确率")
            print("     score: " + str(score))
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
                dimension=e["dimension"],
                label=e["label"]
            )
            if not node.is_leaf:
                for i, j in e["nodes"].items():
                    node.add_node(i, d(j))
            return node
        self.root = d(encode)

    def encode(self):
        def e(node):
            if node is None:
                return None
            node_encode = {
                "is_leaf": node.is_leaf,
                "dimension": node.dimension,
                "label": node.label,
                "nodes": {}
            }
            for i, j in node.nodes.items():
                node_encode["nodes"][i] = e(j)
            return node_encode
        return e(self.root)
