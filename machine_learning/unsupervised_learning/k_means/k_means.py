#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/30 11:32
# @Author   : Mr. Fan

"""

"""

from collections import defaultdict

import numpy
import copy


class KMeans:
    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 聚类数
        self.k = 0
        # 拟合阀值
        self.fit_threshold = 0
        # 迭代次数上限
        self.iteration = 0
        # 样本
        self.sample = None
        # 中心点
        self.centers = {}
        # 聚类
        self.clusters = {}

    def init_parameter(self):
        raw, col = self.sample.shape
        random_index = numpy.random.choice(raw, size=self.k)
        self.centers = [self.sample[i] for i in random_index]
        for i, j in enumerate(self.sample):
            label = self.mark(j)
            if label not in self.clusters:
                self.clusters[label] = [i]
            else:
                self.clusters[label].append(i)
        return

    @staticmethod
    def calculate_distance(x1, x2):
        return sum([(i - j) ** 2 for i, j in zip(x1, x2)])

    def mark(self, p):
        dists = []
        for center in self.centers:
            dists.append(self.calculate_distance(center, p))
        return dists.index(min(dists))

    def update_center(self):
        empty_index = []
        for i, j in self.clusters.items():
            if len(j) == 0:
                empty_index.append(i)
            else:
                self.centers[i] = numpy.mean(self.sample[j], axis=0)
        for i in sorted(empty_index, reverse=True):
            for j in range(i, len(self.clusters) - 1):
                self.clusters[j] = self.clusters[j + 1]
            self.clusters.pop(len(self.clusters) - 1)
            del self.centers[i]
            self.k -= 1
        return

    def update_cluster(self):
        temporary_clusters = copy.deepcopy(self.clusters)
        for label, index in temporary_clusters.items():
            for i in index:
                new_label = self.mark(self.sample[i])
                if new_label == label:
                    continue
                else:
                    self.clusters[label].remove(i)
                    if new_label not in self.clusters:
                        self.clusters[new_label] = [i]
                    else:
                        self.clusters[new_label].append(i)
        return

    def calculate_error(self):
        raw, col = self.sample.shape
        mean_square_error = 0
        for label, index in self.clusters.items():
            partial_data = self.sample[index]
            for p in partial_data:
                mean_square_error += self.calculate_distance(self.centers[label], p)
        return mean_square_error / raw

    def fit(self, sample, k=0, fit_threshold=1, iteration=500):
        sample = numpy.array(sample)

        self.k = k
        self.sample = sample
        self.fit_threshold = fit_threshold
        self.iteration = iteration

        previous_error = float("inf")
        while True:
            self.k += 1
            self.init_parameter()

            last_error = float("inf")
            iteration = 0
            while True:
                iteration += 1
                self.update_center()
                self.update_cluster()
                error = self.calculate_error()

                if self.debug_enable:
                    print("报告: KMeans fit")
                    print("     iteration: " + str(iteration))
                    print("     error:     " + str(error))

                if last_error - error < self.fit_threshold:
                    if self.debug_enable:
                        print("报告: KMeans fit 拟合达到理想状态")
                    break
                else:
                    last_error = error

                if iteration >= self.iteration:
                    if self.debug_enable:
                        print("报告: KMeans fit 迭代完成")
                    break

            if previous_error - error < self.fit_threshold:
                if self.debug_enable:
                    print("报告: KMeans fit 拟合达到理想状态")
                break
            else:
                previous_error = error
        return
