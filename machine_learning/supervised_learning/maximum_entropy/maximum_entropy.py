"""
最大熵模型: 采用IIS最优化算法(M为常数时等同于GIS算法)
"""

from collections import defaultdict

import numpy
import math
import time
import datetime


class MaximumEntropy:
    def __init__(self, debug_enable=False, fit_threshold=0.001, is_fixed=True, max_iteration=10):
        # 调试
        self.debug_enable = debug_enable

        # 拟合阀值
        self.fit_threshold = fit_threshold
        # 迭代次数上限
        self.max_iteration = max_iteration
        # 维度是否固定
        self.is_fixed = is_fixed
        # 加权参数
        self.w = None
        # 标签集合
        self.label_set = None
        # 特征函数
        self.feature = []
        # 边缘分布
        self.marginal_distribution = None
        # 联合分布 特征函数为二值函数 该分布等同于特征的经验期望值
        self.joint_distribution = None
        # 特征期望
        self.expect_feature = None
        # 算法
        self.algorithm = None
        # 训练样本量
        self.raw = 0
        # 训练维度信息
        self.col = None  # 某个训练样本包含特征的总数，这里假设每个样本的M值相同，即M为常数。其倒数类似于学习率

    def fit(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print('------------------------------------------------')
            print('* Error from NaiveBayesianClassifier.fit()    \n'
                  '  The number of samples and tags are different. ')
            print('------------------------------------------------')
            return

        print('[%s] Start training' % datetime.datetime.now())
        if self.debug_enable:
            print("  number of sample: %d" % raw)

        # 生成最大熵模型
        self.initialize(sample, label)

        # 如果样本维数确定
        if self.is_fixed:
            iteration = 0
            begin = time.time()
            while True:
                iteration += 1
                # 使用通用迭代尺度法进行拟合
                delta = self.generalized_iterative_scaling(sample, label)
                # delta绝对值都均小于阈值时 则完成拟合
                if max(abs(delta)) < self.fit_threshold:
                    if self.debug_enable:
                        print('报告: MaximumEntropy fit 拟合完成 残差偏移达到理想值')
                        print("     w: " + str(self.w))
                    break

                self.w += delta

                if iteration >= self.max_iteration:
                    if self.debug_enable:
                        print('报告: MaximumEntropy fit 拟合完成')
                        print("     w: " + str(self.w))
                    break

                if self.debug_enable:
                    print("报告: MaximumEntropy fit 拟合中")
                    print("     iteration: " + str(iteration))
                    print("     time_cost: " + str(time.time() - begin))
                    begin = time.time()

        # 如果样本维度不确定
        else:
            iteration = 0
            delta = numpy.random.rand(len(self.feature))
            while True:
                iteration += 1
                # 使用IIS进行拟合
                delta = self.improved_iterative_scaling(delta, sample, label)
                # if max(abs(delta)) < self.epsilon:
                #     break
                self.w += delta
                if iteration > self.max_iteration:
                    if self.debug_enable:
                        print('报告: MaximumEntropy fit 拟合完成')
                        print("     w: " + str(self.w))
                    break

        print('[%s] Training done' % datetime.datetime.now())

    def initialize(self, sample, label):
        # 根据传入的数据集(数组)初始化模型参数
        self.raw, self.col = sample.shape
        # 边缘分布
        self.marginal_distribution = defaultdict(lambda: 0)
        # 联合分布 特征函数为二值函数 该分布等同于特征的经验期望值
        self.joint_distribution = defaultdict(lambda: 0)
        # 特征期望
        self.expect_feature = defaultdict(lambda: 0)

        label_set = []
        # 生成经验分布模型
        for i, j in zip(sample.tolist(), label.T.tolist()[0]):
            if j not in label_set:
                label_set.append(j)
            i = tuple(i)
            self.marginal_distribution[i] += 1 / self.raw
            self.joint_distribution[(i, j)] += 1 / self.raw
            # 生成特征函数
            for dimension, value in enumerate(i):
                key = (dimension, value, j)
                if key not in self.feature:
                    # 特征函数 （维度，维度下的值，标签）
                    self.feature.append(key)

        self.label_set = tuple(label_set)
        self.w = numpy.zeros(len(self.feature))
        self.generate_expect_feature(sample, label)
        return

    # 计算特征期望
    def generate_expect_feature(self, sample, label):
        # 计算特征的经验期望值
        for i, j in zip(sample.tolist(), label.T.tolist()[0]):
            # 遍历维度 变量值 并标注序号
            for dimension, val in enumerate(i):
                # 获取维度 变量值
                feature = (dimension, val, j)
                # 相关条件分布概率
                self.expect_feature[feature] += self.joint_distribution[(tuple(i), j)]

    # 计算样本条件概率
    def calculate_conditional_probability(self, sample):
        # 当前w下的条件分布概率,输入向量X和y的条件概率
        conditional_probability = defaultdict(float)
        # 遍历标签集
        for i in self.label_set:
            index = 0
            for dimension, value in enumerate(sample):
                temporary_feature = (dimension, value, i)
                # 特征函数匹配
                if temporary_feature in self.feature:
                    index += self.w[self.feature.index(temporary_feature)]
            conditional_probability[i] = math.exp(index)
        normalizer = sum(conditional_probability.values())
        for key, value in conditional_probability.items():
            # 条件概率最小为 1 / 总条件概率
            conditional_probability[key] = value / normalizer
        return conditional_probability

    def get_expect_feature(self, sample, label):
        # 基于当前模型，获取每个特征估计期望
        expect_feature = defaultdict(float)
        iteration = 0
        for i, j in zip(sample.tolist(), label.T.tolist()[0]):
            print(iteration, self.raw)
            iteration += 1
            conditional_probability = self.calculate_conditional_probability(i)[j]
            for dimension, value in enumerate(i):
                expect_feature[(dimension, value, j)] += self.marginal_distribution[tuple(i)] * conditional_probability
        return expect_feature

    def generalized_iterative_scaling(self, sample, label):
        expect_feature = self.get_expect_feature(sample, label)
        delta = numpy.zeros(len(self.feature))
        for j in range(len(self.feature)):
            delta[j] = 1 / self.col * math.log(self.expect_feature[self.feature[j]] / expect_feature[self.feature[j]])
        # 归一化，防止某特征权重过大超出计算范围
        delta = delta / delta.sum()
        return delta

    def improved_iterative_scaling(self, delta, X_data, y_data):
        """
        改进迭代尺度法
        Parameters
        ----------
        delta
        X_data
        y_data

        Returns
        -------

        """
        # IIS算法更新delta
        g = numpy.zeros(len(self.feature))
        g_diff = numpy.zeros(len(self.feature))
        for j in range(len(self.feature)):
            for k in range(self.raw):
                g[j] += self.marginal_distribution[tuple(X_data[k])] * \
                        self.calculate_conditional_probability(X_data[k])[y_data[k]] * \
                        math.exp(delta[j] * self.col[k])
                g_diff[j] += g[j] * self.col[k]
            g[j] -= self.expect_feature[j]
            delta[j] -= g[j] / g_diff[j]
        return delta

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            conditional_probability = self.calculate_conditional_probability(sample[i, :].tolist()[0])
            if self.debug_enable:
                print("报告: MaximumEntropy predict 预测条件概率")
                print("     conditional_probability: " + str(conditional_probability))
            label[i, 0] = max(conditional_probability, key=conditional_probability.get)
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
        right_count = 0
        for i in range(raw):
            if label_predict[i, 0] == label[i, 0]:
                right_count += 1
        return right_count / raw
