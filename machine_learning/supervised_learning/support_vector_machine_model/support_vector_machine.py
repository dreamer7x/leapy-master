import math
import random

import numpy


class SupportVectorMachine:

    def __init__(self, debug_enable=False):
        # 调试
        self.debug_enable = debug_enable

        # 迭代次数上限
        self.iteration = 0
        # 学习速度
        self.learning_rate = 0
        # 核函数
        self.kernel = None
        # 优化算法
        self.algorithm = None
        # 容错率
        self.tolerance = 0
        # 拟合阈值
        self.fit_threshold = 0
        # 惩罚项
        self.c = 0
        # 一般参数
        self.p = None
        # 训练样本
        self.sample = None
        # 训练标签
        self.label = None
        # 累计参数
        self.a = None
        # 核函数列表
        self.k = None
        # 超平面变量系数
        self.w = None
        # 超平面高度系数
        self.b = None
        # 缓冲误差
        self.e = None

    # 计算核函数
    def get_kernel(self, x1, x2):
        if self.kernel == "linear_kernel_function":
            x1 = numpy.mat(x1)
            x2 = numpy.mat(x2)
            raw1, col1 = x1.shape
            raw2, col2 = x2.shape
            k = numpy.mat(numpy.zeros((raw1, raw2)))
            for i in range(raw1):
                for j in range(raw2):
                    k[i, j] = x1[i, :] * x2[j, :].T
            return k

        # 多项式核函数
        # p[0] 表示多项式次数
        if self.kernel == "polynomial_kernel_function":
            x1 = numpy.mat(x1)
            x2 = numpy.mat(x2)
            raw1, col1 = x1.shape
            raw2, col2 = x2.shape
            k = numpy.mat(numpy.zeros((raw1, raw2)))
            for i in range(raw1):
                for j in range(raw2):
                    k[i, j] = pow(x1[i, :] * x2[j, :].T + 1, self.p[0])
            return k

        # 高斯核函数
        # p[0] 高斯公式分母
        if self.kernel == "gaussian_kernel_function":
            x1 = numpy.mat(x1)
            x2 = numpy.mat(x2)
            raw1, col1 = x1.shape
            raw2, col2 = x2.shape
            k = numpy.mat(numpy.zeros((raw1, raw2)))
            for i in range(raw1):
                for j in range(raw2):
                    temporary_x = x1[i, :] - x2[j, :]
                    k[i, j] = math.exp(-temporary_x * temporary_x.T / (2 * (self.p[0] ** 2)))
            print(k)
            return k

        # 字符串核函数
        if self.kernel == "string_kernel_function":
            return None

    def get_j(self, i):

        if self.algorithm is None:
            j = i
            while j == i:
                j = int(random.uniform(0, self.sample.shape[0]))
            return j, self.get_e(j)

        elif self.algorithm == "sequential_minimal_optimization" or \
                self.algorithm == "SMO" or \
                self.algorithm == "smo":
            j = i
            while j == i:
                j = int(random.uniform(0, self.sample.shape[0]))
            return j, self.get_e(j)

    # 计算误差
    def get_e(self, i):
        f = numpy.multiply(self.a, self.label).T * self.k[:, i] + self.b
        e = f - self.label[i, 0]
        return e

    def get_g(self, k):
        return numpy.multiply(self.a, self.label).T * k + self.b

    @staticmethod
    def clip_a(a, h, l):
        if a > h:
            a = h
        if l > a:
            a = l
        return a

    def sequential_minimal_optimization(self):
        # 样本量 特征量
        raw, col = self.sample.shape
        # 初始化 alpha 参数
        self.a = numpy.mat(numpy.zeros((raw, 1)))
        # 超平面高度参数
        self.b = 0
        # 累计误差
        self.e = numpy.mat(numpy.zeros((raw, 1)))
        # 核函数
        self.k = self.get_kernel(self.sample, self.sample)

        # 进行外循环
        iteration = 0
        # 进行有限次迭代
        while iteration < self.iteration:
            # 调整频率
            change_time = 0
            # 外层循环
            for i in range(raw):
                # 检验是否满足约束条件 该步骤十分巧妙将约束条件代入考虑范围
                ei = self.get_e(i)
                # 如果满足外层循环启发式规则 该判定为KKT条件的一个变换
                if (self.label[i, 0] * ei < -self.tolerance and self.a[i] < self.c) or \
                        (self.label[i, 0] * ei > self.tolerance and self.a[i] > 0):
                    j, ej = self.get_j(i)
                    old_ai = self.a[i].copy()
                    old_aj = self.a[j].copy()

                    if self.label[i, 0] != self.label[j, 0]:
                        l = max(0, self.a[j, 0] - self.a[i, 0])
                        h = min(self.c, self.c + self.a[j, 0] - self.a[i, 0])
                    else:
                        l = max(0, self.a[j, 0] + self.a[i, 0] - self.c)
                        h = min(self.c, self.a[j, 0] + self.a[i, 0])

                    # 如果 下限 == 上限
                    if l == h:
                        # print("报告: SupportVectorMachine sequential_minimal_optimization aj 变化小于阈值")
                        continue

                    step = self.k[i, i] + self.k[j, j] - 2.0 * self.k[i, j]

                    if step <= 0:
                        continue

                    # 步骤4：更新 aj
                    self.a[j, 0] += self.label[j, 0] * (ei - ej) / step

                    # 步骤5：修剪 aj
                    self.a[j, 0] = self.clip_a(self.a[j, 0], h, l)
                    # 更新 ej 至误差缓存
                    self.e[j, 0] = self.get_e(j)

                    if abs(self.a[j] - old_aj) < self.fit_threshold:
                        # print("报告: SupportVectorMachine sequential_minimal_optimization aj 变化小于阈值")
                        continue

                    # 步骤6：更新 ai
                    self.a[i] += self.label[j, 0] * self.label[i, 0] * (old_aj - self.a[j, 0])

                    # 步骤7：更新 b1 和 b2
                    b1 = self.b - ei - \
                        self.label[i, 0] * (self.a[i, 0] - old_ai) * self.sample[i, :] * self.sample[i, :].T - \
                        self.label[j, 0] * (self.a[j, 0] - old_aj) * self.sample[i, :] * self.sample[j, :].T
                    b2 = self.b - ej - \
                        self.label[i, 0] * (self.a[i, 0] - old_ai) * self.sample[i, :] * self.sample[j, :].T - \
                        self.label[j, 0] * (self.a[j, 0] - old_aj) * self.sample[j, :] * self.sample[j, :].T
                    if 0 < self.a[i] < self.c:
                        self.b = b1
                    elif 0 < self.a[j] < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    change_time += 1
            print("报告: SupportVectorMachine sequential_minimal_optimization 迭代次数: " + str(iteration))
            iteration += 1

    def fit(self, sample, label, iteration=100, learning_rate=0.01, tolerance=0.001, fit_threshold=0.0001, c=0.5,
            kernel=None, algorithm=None, p=None):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print("错误: Classification fit 样本数，标签数不等")
            return

        self.sample = sample
        self.label = label
        self.iteration = iteration
        self.learning_rate = learning_rate
        if kernel is None:
            self.kernel = "linear_kernel_function"
        else:
            self.kernel = kernel
        if algorithm is None:
            self.algorithm = "sequential_minimal_optimization"
        else:
            self.algorithm = algorithm
        self.tolerance = tolerance
        self.fit_threshold = fit_threshold
        self.c = c
        self.p = p

        # 默认采用序列最小最优化算法
        if self.algorithm is None or \
                self.algorithm == "sequential_minimal_optimization" or \
                self.algorithm == "SMO" or \
                self.algorithm == "smo":
            # 拟合模型
            self.sequential_minimal_optimization()
            # 超平面变量参数
            self.w = self.get_w()
            return

    def get_w(self):
        raw, col = numpy.shape(self.sample)
        w = numpy.mat(numpy.zeros((col, 1)))
        for i in range(raw):
            w += numpy.multiply(self.a[i, 0] * self.label[i, 0], self.sample[i, :].T)
        return w

    def predict(self, sample):
        if self.kernel is None or self.kernel == "linear_kernel_function":
            sample = numpy.mat(sample)
            raw, col = sample.shape
            label = numpy.mat(numpy.empty((raw, 1)))
            for i in range(raw):
                if sample[i, :] * self.w + self.b > 0:
                    label[i, 0] = 1
                else:
                    label[i, 0] = -1
            return label
        else:
            sample = numpy.mat(sample)
            raw, col = sample.shape
            k = self.get_kernel(self.sample, sample)
            label = numpy.mat(numpy.empty((raw, 1)))
            for i in range(raw):
                if self.get_g(k[:, i]) > 0:
                    label[i, 0] = 1
                else:
                    label[i, 0] = -1
            return label

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        label_predict = self.predict(sample)

        right_count = 0
        for i in range(raw):
            if label_predict[i, 0] == label[i, 0]:
                right_count += 1

        return right_count / raw
