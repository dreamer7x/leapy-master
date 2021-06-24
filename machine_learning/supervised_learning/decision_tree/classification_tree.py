#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/23 15:11
# @Author   : Mr. Fan

"""分类树模块

分类树算法模块，解决多分类问题，支持离散属性，通过连续离散化支持连续属性。

Classes
-------
ClassificationTree : 分类树类对象
"""

import numpy
import pandas
import json


class ClassificationTree:
    def __init__(self, debug_enable=False):
        """
        初始化分类树

        Parameters
        ----------
        debug_enable : bool
            调试
        """
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
        # 属性分类权重
        self.weights = None

    def fit(self, sample, label, fit_threshold=0.01, penalty_term=1,
            dispersed=None, max_deep=20, weights=None):
        """
        拟合

        Parameters
        ----------
        sample : matrix N * p
            样本，N行样本，p列属性
        label : matrix N * 1
            标签，列向量
        dispersed : list p
            离散化信息
        max_deep : int
            最大深度
        fit_threshold : float
            拟合阀值
        penalty_term : float
            惩罚项
        weights : list p
            属性分类权重
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: Classification fit 样本数，标签数不等")
            return

        # 如果离散化信息未提供 则设为默认离散
        if dispersed is None or dispersed is True:
            dispersed = [True] * sample.shape[1]
        if dispersed is False:
            dispersed = [False] * sample.shape[1]
        if col != len(dispersed):
            print("错误: Classification fit 属性数，离散化信息数不等")
            return

        if weights is None:
            weights = [1] * col
        if len(weights) != col:
            print("错误: Classification fit 属性数，属性分类权重数不等")
            return
        self.weights = weights

        self.dispersed = dispersed
        self.max_deep = max_deep
        self.fit_threshold = fit_threshold
        self.penalty_term = penalty_term

        self.root = self.classification_tree(sample, label)

    class TreeNode:
        def __init__(self, is_leaf=True, is_dispersed=True, dimension=None, value=None, label=None):
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
            # 子节点
            self.nodes = {}

        def add_node(self, value, node):
            """
            添加结点

            Parameters
            ----------
            value : ...
                键
            node : TreeNode
                结点类对象
            """
            self.nodes[value] = node

        def predict(self, sample):
            """
            预测

            Parameters
            ----------
            sample : matrix 1 * p
                单样本

            Returns
            -------
            predict : ...
                标签
            """
            # 如果为叶节点
            if self.is_leaf is True:
                return self.label
            # 如果为离散值结点
            if self.is_dispersed:
                if sample[0, self.dimension] == self.value:
                    return self.nodes['left'].forward(sample, )
                else:
                    return self.nodes['right'].forward(sample, )
            else:
                if sample[0, self.dimension] <= self.value:
                    return self.nodes['left'].forward(sample, )
                else:
                    return self.nodes['right'].forward(sample, )

    @staticmethod
    def gini_index(label):
        """
        计算基尼指数

        Parameters
        ----------
        label : list
            标签列表。

        Returns
        -------
        index : float
            标签列表的基尼指数。
        """
        label_count = pandas.value_counts(label)
        sample_size = sum(label_count.values)
        index = 1 - sum([pow(i / sample_size, 2) for i in label_count.values])
        return index

    def conditional_gini_index(self, sample, label, is_dispersed=True, axis=0):
        """
        计算特定属性列的条件基尼指数，并获取最优切分点

        Parameters
        ----------
        sample : matrix N * p
            样本矩阵，N行样本，p列维度
        label : matrix N * 1
            标签矩阵，列向量
        is_dispersed : bool
            是否为离散属性，False则进行离散化处理
        axis : int
            属性列

        Returns
        -------
        value : float
            最优切分点
        min_index : float
            最优基尼指数
        """
        raw, col = sample.shape
        label_split = {}
        for i in range(raw):
            # 如果切分点不存在，则创建切分点
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
        # 遍历计算基尼指数
        index = []
        for d in label_split.values():
            left_gini = len(d[0]) / raw * self.gini_index(d[0])
            right_gini = len(d[1]) / raw * self.gini_index(d[1])
            index.append(left_gini + right_gini)
        min_index = min(enumerate(index), key=lambda f: f[1])
        value = list(label_split.keys())[min_index[0]]
        return value, min_index[1]

    @staticmethod
    def split_data(sample, label, dimension, value, is_dispersed=True):
        """
        切割数据

        Parameters
        ----------
        sample : matrix N * p
            样本，N行样本，p列属性
        label : matrix N * 1
            标签，列向量
        dimension : int
            维度
        value : ...
            切割值
        is_dispersed : bool
            是否为离散属性

        Returns
        -------
        left_sample, left_label, right_sample, right_label : matrix N * v
            切割数据
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

    def gain_dimension(self, sample, label):
        """
        计算最佳切割位置

        Parameters
        ----------
        sample : matrix N * p
            样本，N行样本，p列属性
        label : matrix N * 1
            标签，列向量

        Returns
        -------
        gain: (dimension, value, gini_index)
            最佳切割位置信息
            dimension : 切割维度
            value : 切割点
            gini_index : 基尼指数
        """
        raw, col = sample.shape
        gini_index = []
        for i in range(col):
            if self.dispersed[i]:
                value = self.conditional_gini_index(sample, label, is_dispersed=True, axis=i)
                # 由于这里取最小值，所以除以对应的属性权重
                gini_index.append([i, value[0], value[1] / self.weights[i]])
            else:
                value = self.conditional_gini_index(sample, label, is_dispersed=False, axis=i)
                # 由于这里取最小值，所以除以对应的属性权重
                gini_index.append((i, value[0], value[1] / self.weights[i]))
        gain = sorted(gini_index, key=lambda f: f[-1])[0]
        return gain

    # CT
    def classification_tree(self, sample, label, deep=1):
        """
        构造分类树

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签
        deep : int
            深度

        Returns
        -------
        node : TreeNode
            树结点
        """
        raw, col = sample.shape
        label_count = pandas.value_counts(label.T.tolist()[0])

        # 1 如果只有只用类别，则返回叶结点
        if label_count.shape[0] == 1:
            return self.TreeNode(is_leaf=True,
                                 label=label[0, 0])

        # 2 如果只有一种特性，则返回叶节点
        if col == 0:
            node_label = sorted(label_count.items(), key=lambda x: x[1])[-1][0]
            return self.TreeNode(is_leaf=True,
                                 label=node_label)

        # - 若达到深度上限
        if deep >= self.max_deep:
            return self.TreeNode(is_leaf=True,
                                 label=max(label_count.items(), key=lambda x: x[-1])[0])

        # 3 获取基尼指数最大特征
        dimension, value, min_gini_index = self.gain_dimension(sample, label)

        # 4 如果基尼指数小于拟合阈值
        if min_gini_index < self.fit_threshold:
            return self.TreeNode(is_leaf=True,
                                 label=max(label_count.items(), key=lambda x: x[-1])[0])

        # - 删除属性列
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
        node.add_node("left", self.classification_tree(left_sample, left_label, deep=deep + 1))
        node.add_node("right", self.classification_tree(right_sample, right_label, deep=deep + 1))

        return node

    def get_leaf(self, node):
        """
        获取树结点下叶结点数量

        Parameters
        ----------
        node : TreeNode
            树结点

        Returns
        -------
        count: int
            叶节点数量
        """
        if node.is_leaf:
            return 1
        count = 0
        for i in node.nodes.keys():
            count += self.get_leaf(node.nodes[i])
        return count

    # 计算损失函数
    @staticmethod
    def get_error(label_predict, label):
        """
        计算 0-1 损失函数

        Parameters
        ----------
        label_predict : int
            预测标签
        label : matrix N * 1
            标签

        Returns
        -------
        error: int
            错误数量
        """
        raw, col = label.shape
        error = 0
        for i in range(raw):
            if label[i, 0] != label_predict:
                error += 1
        return error

    def prune_classification_tree(self, node, sample, label, dispersed):
        """
        剪枝

        Parameters
        ----------
        node : TreeNode
            结点
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签
        dispersed : list
            离散化信息

        Returns
        -------
        node : TreeNode
            返回剪枝后结点
        """
        raw, col = sample.shape
        # 如果验证样本为空 进行塌陷处理
        if raw == 0:
            labels = self.get_leaf(node)
            return self.TreeNode(is_leaf=True,
                                 label=max(pandas.value_counts(labels), key=lambda x: x[-1])[0])
        # 如果为叶节点
        if not node.is_leaf:
            left_sample, left_label, right_sample, right_label = \
                self.split_data(sample, label, node.dimension, node.value, is_dispersed=self.dispersed[node.dimension])
            left_node = self.prune_classification_tree(node.nodes['left'], left_sample, left_label, dispersed)
            right_node = self.prune_classification_tree(node.nodes['right'], right_sample, right_label, dispersed)
            if left_node.is_leaf and right_node.is_leaf:
                # 这里采用 0-1 损失函数
                # 计算叶结点损失数
                error_leaf = 0
                error_leaf += self.get_error(left_node.label_set, left_label)
                error_leaf += self.get_error(right_node.label_set, right_label)

                # 计算根节点损失数
                tree_label = max(pandas.value_counts(label.T.tolist()[0]), key=lambda x: x[-1])[0]
                error_tree = self.get_error(tree_label, label)

                if error_tree < error_leaf + self.penalty_term:
                    return self.TreeNode(is_leaf=True,
                                         label=tree_label)
                else:
                    return node
            else:
                return node
        else:
            return node

    def test(self, sample, label, dispersed):
        """
        验证

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签
        dispersed : list p列
            离散化信息
        """
        sample = numpy.mat(sample)
        label = numpy.mat(label)
        self.root = self.prune_classification_tree(self.root, sample, label, dispersed)

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
            预测标签
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label_predict = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            label_predict[i, 0] = self.root.forward(sample[i, :], )
        return label_predict

    def score(self, sample, label):
        """
        评估

        Parameters
        ----------
        sample : matrix N * p
            样本
        label : matrix N * 1
            标签

        Returns
        -------
        score: float
            正确率
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(label).T
        label_predict = self.predict(sample)
        count = 0
        for i in range(raw):
            if label[i, 0] == label_predict[i, 0]:
                count += 1
        score = count / raw
        return score

    def load(self, path):
        file = open(path, "r")
        model_encode = json.load(file)
        self.decode(model_encode)

    def save(self, path):
        file = open(path, "w")
        model_encode = self.encode()
        json.dump(model_encode, file)

    def decode(self, encode):
        def d(e):
            node = self.TreeNode(
                is_leaf=e["is_leaf"],
                is_dispersed=e["is_dispersed"],
                dimension=e["is_dispersed"],
                value=e["value"]
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
                "left": e(node.left),
                "right": e(node.right)
            }
            return encode
        return e(self.root)
