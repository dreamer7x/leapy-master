#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/30 8:05
# @Author   : Mr. Fan

"""
聚合聚类

"""

import numpy
import math


class HierarchicalClustering:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 聚类数据样本
        self.sample = None
        # 聚类列表 [hierarchical_node, ...]
        self.clusters = None
        # 聚合最大距离
        self.max_distance = 0
        # 分裂最小距离
        self.min_distance = 0

    class hierarchical_node:
        def __init__(self, center, left=None, right=None, flag=None, distance=0, weight=1):
            # 中心
            self.center = center
            # 左结点
            self.left = left
            # 右结点
            self.right = right
            # 标号 提供样本索引信息
            self.flag = flag
            # 距离
            self.distance = distance
            # 权重 一般为结点数
            self.weight = weight

        def traverse(self, node=None):
            if node is None:
                node = self
            if node.left is None and node.right is None:
                return [node.center.tolist()]
            else:
                return self.traverse(node.left) + self.traverse(node.right)

    @staticmethod
    def euclidean_distance(x1, x2):
        distance = math.sqrt(numpy.sum((x1 - x2) ** 2))
        return distance

    def fit(self, sample, n, algorithm=None, max_distance=1):
        sample = numpy.array(sample)
        self.sample = sample
        self.max_distance = max_distance

        if algorithm is None:
            algorithm = "aggregation_clustering"

        if algorithm == "aggregation_clustering":
            self.aggregation_clustering(n)

    def aggregation_clustering(self, n, distance=None):
        if distance is None:
            distance = "euclidean_distance"

        raw, col = self.sample.shape

        # 初始聚类模型
        clusters = [self.hierarchical_node(self.sample[i], flag=i) for i in range(raw)]
        # 构造距离映射
        distances = {}
        min_flag1 = None
        min_flag2 = None
        is_influence = True
        current_cluster = len(clusters)

        while True:
            min_distance = float("inf")

            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    if distances.get((clusters[i].flag, clusters[j].flag)) is None:
                        if distance == "euclidean_distance":
                            distances[(clusters[i].flag, clusters[j].flag)] = \
                                self.euclidean_distance(clusters[i].center, clusters[j].center)

                    if distances[(clusters[i].flag, clusters[j].flag)] <= min_distance:
                        min_flag1 = i
                        min_flag2 = j
                        min_distance = distances[clusters[i].flag, clusters[j].flag]
                        if min_distance >= self.max_distance:
                            is_influence = False

            if min_flag1 is not None and min_flag2 is not None and min_distance != float("inf"):
                if is_influence:
                    center = (clusters[min_flag1].center + clusters[min_flag2].center) / 2
                else:
                    if clusters[min_flag1].weight == clusters[min_flag2].weight:
                        if self.debug_enable and clusters[min_flag1].weight > raw // n:
                            print("警告: HierarchicalClustering aggregation_clustering 出现等权重类")
                        center = (clusters[min_flag1].center + clusters[min_flag2].center) / 2
                    elif clusters[min_flag1].weight > clusters[min_flag2].weight:
                        center = clusters[min_flag1].center
                    else:
                        center = clusters[min_flag2].center
                    is_influence = True
                flag = current_cluster
                weight = clusters[min_flag1].weight + clusters[min_flag2].weight
                current_cluster += 1
                cluster = self.hierarchical_node(center, clusters[min_flag1], clusters[min_flag2],
                                                 flag, min_distance, weight)
                del clusters[min_flag2]
                del clusters[min_flag1]
                clusters.append(cluster)

            if len(clusters) <= n:
                if self.debug_enable:
                    print("报告: HierarchicalClustering aggregation_clustering 聚类完成")
                    print("     cluster: ")
                    for i in clusters:
                        print("              " + str(i.center.tolist()))
                break

        self.clusters = clusters

    def get_area(self):
        area = []
        for i in self.clusters:
            area.append(numpy.array(i.traverse()))
        return area

    def get_center(self):
        center = []
        for i in self.clusters:
            center.append(i.center.tolist())
        center = numpy.array(center)
        return center
