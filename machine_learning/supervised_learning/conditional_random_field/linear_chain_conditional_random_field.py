#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/28 8:08
# @Author   : Mr. Fan

"""
条件随机场

包括 :
    线性链条件随机场
"""

from collections import Counter
from scipy.optimize import fmin_l_bfgs_b

import numpy
import time
import math
import json
import datetime


# 标记 t = -1
FIRST_LABEL = '*'
# 标记索引
FIRST_LABEL_INDEX = 0
# 拟合阀值
SCALING_THRESHOLD = 1e250
# 迭代次数
ITERATION = 0
# 最大迭代次数
SUB_ITERATION = 0
# 总迭代次数
TOTAL_ITERATION = 0
# 重力系数
GRADIENT = 0


class LinearChainConditionalRandomField:

    def __init__(self, debug_enable=False):
        self.debug_enable = debug_enable

        # 训练数据
        self.data = None
        # 特征模型 Feature 类对象
        self.feature = None
        # 类别映射 类型 -> 类型标号
        self.label_map = None
        # 类别列表
        self.label = None
        # 类别数
        self.label_number = None
        # 模型参数
        self.parameter = None
        # L-BFGS 参数
        self.sigma = 0

    @staticmethod
    def generate_potential_table(parameter, label_number, feature, sample, inference=True):
        """
        为每个时序创建势函数表，用以支持前向后向算法进行概率计算，用以支持维特比算法进行观测数据生成

        Parameters
        ----------
        parameter : dict
            参数列表
        label_number : int
            状态数
        feature : Feature
            特征模型
        sample : list / dict
            观测序列 / 特定时序的特征函数集合
        inference : bool
            表示特征函数矩阵生成 具有多种状态
            True : sample list 为观测序列
            False : sample dict 为特定时序的特征函数集合

        Returns
        -------
        list
            势函数列表
            potential_table[t][previous_label, label]
            = exp(conditional_probability(params, feature_vector(previous_label, label, sample, t)))
            (where 0 <= t < len(sample))
        """
        potential_table = []
        for t in range(len(sample)):
            # 以下将统计加和转换为矩阵形式
            table = numpy.zeros((label_number, label_number))
            # 如果 sample 为观测序列
            if inference:
                for (previous_label, label), probability in \
                        feature.calculate_conditional_probability(parameter, sample, t):
                    # 如果为状态特征函数
                    if previous_label == -1:
                        table[:, label] += probability
                    # 如果为转移特征函数
                    else:
                        table[previous_label, label] += probability
            # 如果 sample 为特征模型
            else:
                for (previous_label, label), feature_id in sample[t]:
                    probability = sum(parameter[i] for i in feature_id)
                    if previous_label == -1:
                        table[:, label] += probability
                    else:
                        table[previous_label, label] += probability
            table = numpy.exp(table)
            if t == 0:
                table[FIRST_LABEL_INDEX + 1:] = 0
            else:
                table[:, FIRST_LABEL_INDEX] = 0
                table[FIRST_LABEL_INDEX, :] = 0
            potential_table.append(table)

        return potential_table

    @staticmethod
    def forward_backward(label_number, time_length, potential_table):
        """
        前向，后向算法，从而计算观测序列在某一特定时序为 i 状态的概率
        with a scaling method(suggested by Rabiner, 1989).

        Parameters
        ----------
        label_number : int
            标签数量
        time_length : int
            时序长度
        potential_table : list
            势函数列表

        Returns
        -------
        alpha : M * N
            前向概率矩阵
            M : 时序长度
            N : 状态数
        beta : M * N
            后向概率矩阵
            M : 时序长度
            N : 状态数
        probability : float
            时序概率
        scaling_map : dict
            比例项

        * Reference:
            - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
            in Speech Recognition
        """
        potential_table = list(potential_table)
        # t * n
        alpha = numpy.zeros((time_length, label_number))
        scaling_map = dict()
        t = 0
        for label_id in range(label_number):
            alpha[t, label_id] = potential_table[t][FIRST_LABEL_INDEX, label_id]
        # alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :] 低效率
        t = 1
        while t < time_length:
            scaling_coefficient = None
            is_overflow = False
            label_id = 1
            while label_id < label_number:
                alpha[t, label_id] = numpy.dot(alpha[t - 1, :], potential_table[t][:, label_id])
                # 如果某一状态发生概率超过比例阀值
                if alpha[t, label_id] > SCALING_THRESHOLD:
                    if is_overflow:
                        print("错误: LinearChainConditionalRandomField forward_backward 连续溢出")
                        raise BaseException()
                    is_overflow = True
                    scaling_time = t - 1
                    scaling_coefficient = SCALING_THRESHOLD
                    scaling_map[scaling_time] = scaling_coefficient
                    break
                else:
                    label_id += 1
            if is_overflow:
                alpha[t - 1] /= scaling_coefficient
                alpha[t] = 0
            else:
                t += 1

        beta = numpy.zeros((time_length, label_number))
        t = time_length - 1
        for label_id in range(label_number):
            beta[t, label_id] = 1.0
        # beta[time_length - 1, :] = 1.0 低效率
        for t in range(time_length - 2, -1, -1):
            for label_id in range(1, label_number):
                beta[t, label_id] = numpy.dot(beta[t + 1, :], potential_table[t + 1][label_id, :])
            # 如果时序出现在异常字典中 则进行等比例处理
            if t in scaling_map.keys():
                beta[t] /= scaling_map[t]

        probability = sum(alpha[time_length - 1])

        return alpha, beta, probability, scaling_map

    @staticmethod
    def calculate_score(potential_table, scaling_map, label, label_map):
        """
        计算预测分数

        Parameters
        ----------
        potential_table :
            势函数列表
        scaling_map : dict
            比例项
        label :
            状态标号
        label_map : dict
            状态映射

        Returns
        -------
        float
            预测分数
        """
        score = 1.0
        previous_label = FIRST_LABEL_INDEX
        for t in range(len(label)):
            label = label_map[label[t]]
            score *= potential_table[previous_label, label, t]
            if t in scaling_map.keys():
                score = score / scaling_map[t]
            previous_label = label
        return score

    def log_likelihood(self, parameter, *args):
        """
        计算对数似然函数

        Parameters
        ----------
        parameter : dict
            模型参数
        args : tuple
            参数元组

        Returns
        -------
        float
            似然函数
        """
        data, feature_model, feature_data, empirical_counts, label_map, squared_sigma = args
        expected_count = numpy.zeros(len(feature_model))

        total_log_probability = 0
        for features in feature_data:
            potential_table = self.generate_potential_table(parameter, len(label_map), feature_model,
                                                            features, inference=False)
            potential_table = list(potential_table)
            alpha, beta, probability, scaling_map = \
                self.forward_backward(len(label_map), len(features), potential_table)
            total_log_probability += math.log(probability) + \
                sum(math.log(scaling_coefficient) for _, scaling_coefficient in scaling_map.items())
            for t in range(len(features)):
                potential = potential_table[t]
                for (previous_label, label), feature_ids in features[t]:
                    # p(prev_y, y | X, t)
                    if previous_label == -1:
                        if t in scaling_map.keys():
                            prob = (alpha[t, label] * beta[t, label] * scaling_map[t]) / probability
                        else:
                            prob = (alpha[t, label] * beta[t, label]) / probability
                    elif t == 0:
                        if previous_label is not FIRST_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[FIRST_LABEL_INDEX, label] * beta[t, label]) / probability
                    else:
                        if previous_label is FIRST_LABEL_INDEX or label is FIRST_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t - 1, previous_label] *
                                    potential[previous_label, label] *
                                    beta[t, label]) / probability
                    for fid in feature_ids:
                        expected_count[fid] += prob

        likelihood = numpy.dot(empirical_counts, parameter) - total_log_probability - \
            numpy.sum(numpy.dot(parameter, parameter)) / (squared_sigma * 2)

        gradients = empirical_counts - expected_count - parameter / squared_sigma
        global GRADIENT
        GRADIENT = gradients

        global SUB_ITERATION

        if self.debug_enable:
            sub_iteration_str = '    '
            if SUB_ITERATION > 0:
                sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION) + ')'
            print('   ', '{0:03d}'.format(ITERATION), sub_iteration_str, ':', likelihood * -1)

        SUB_ITERATION += 1

        return -1 * likelihood

    @staticmethod
    def gradient(parameter, *args):
        global GRADIENT
        return -1 * GRADIENT

    @staticmethod
    def callback(parameter):
        global ITERATION
        global TOTAL_ITERATION
        global SUB_ITERATION
        ITERATION += 1
        TOTAL_ITERATION += SUB_ITERATION
        SUB_ITERATION = 0

    @staticmethod
    def read_file(filename):
        """
        分词文件读取接口

        Parameters
        ----------
        filename : str
            文件索引

        Returns
        -------
        list
            数据
        """
        data = list()
        data_string_list = list(open(filename))

        element_size = 0
        sample = list()
        label = list()
        for data_string in data_string_list:
            words = data_string.strip().split()
            if len(words) == 0:
                data.append((sample, label))
                sample = list()
                label = list()
            else:
                if element_size == 0:
                    element_size = len(words)
                elif element_size is not len(words):
                    raise FileNotFoundError
                sample.append(words[:-1])
                label.append(words[-1])
        if len(sample) > 0:
            data.append((sample, label))

        return data

    def get_feature(self):
        """
        获取观测序列在各个时序的特征函数集合

        Returns
        -------
        list
            特征函数集合列表
        """
        return [[self.feature.get_feature_list(sample, t) for t in range(len(sample))]
                for sample, _ in self.data]

    def quasi_newton(self):
        """
        拟合参数 同时显示过程数据

        L-BFGS

        References
        ----------
        [1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization,
            (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
        [2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large
            scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4,
            pp. 550 - 560.
        [3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
            large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        """

        feature_data = self.get_feature()
        print('* start quasi_newton L-BFGS')
        if self.debug_enable:
            print('  squared sigma:', self.sigma)
            print('============================================')
            print('    iteration : likelihood   ')
            print('--------------------------------------------')
        self.parameter, log_likelihood, information = \
            fmin_l_bfgs_b(func=self.log_likelihood, fprime=self.gradient,
                          x0=numpy.zeros(len(self.feature)),
                          args=(self.data, self.feature, feature_data,
                                self.feature.get_empirical_count(),
                                self.label_map, self.sigma),
                          callback=self.callback)
        if self.debug_enable:
            print('============================================')
            print('* training finished')
            print('  iteration: ' + str(information['nit']))
            print('  likelihood: %s' % str(log_likelihood))
            if information['warnflag'] != 0:
                print('  warning code: %d' % information['warnflag'])
                if 'task' in information.keys():
                    print('  reason: %s' % (information['task']))

    def fit(self, data=None, data_path=None, model_path=None, sigma=10.0, algorithm=None):
        """
        训练模型

        Parameters
        ----------
        data
        data_path
        model_path
        sigma
        algorithm

        Returns
        -------

        """
        if algorithm is None:
            algorithm = "quasi_newton"

        self.sigma = sigma
        self.data = data

        start_time = time.time()

        print('[%s] start training' % datetime.datetime.now())
        if data_path is not None:
            print("* reading training handwritten_numeral ... ", end="")
            # 获取训练数据
            self.data = self.read_file(data_path)
            print("done")
        # 特征统计模型
        self.feature = self.Feature()
        # 使用特征统计模型扫描训练数据
        self.feature.scan(self.data)
        # 获取标签字典，标签集合
        self.label_map, self.label = self.feature.get_label()
        # 状态列表
        self.label = list(self.label)
        # 标签数量
        self.label_number = len(self.label)

        if self.debug_enable:
            print("  number of label: %d" % (self.label_number - 1))
            print("  number of feature: %d" % len(self.feature))

        if algorithm == "quasi_newton":
            # 拟牛顿法
            self.quasi_newton()

        if model_path is not None:
            # 保存模型
            self.save(model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.debug_enable:
            print('  elapsed time: %f' % elapsed_time)
        print('[%s] Training done' % datetime.datetime.now())

    def score(self, data=None, data_path=None, model_path=None):
        print('[%s] Start testing' % datetime.datetime.now())
        # 验证
        if model_path is not None:
            self.load(model_path)
        if self.feature is None:
            print('[%s] Testing fail : You should load a model first' % datetime.datetime.now())
            return
        test_data = data
        if data_path is not None:
            test_data = self.read_file(data_path)
        if test_data is None:
            print('[%s] Testing fail : Test data acquisition failed' % datetime.datetime.now())
            return

        total_count = 0
        correct_count = 0
        for sample, label in test_data:
            label_predict = self.predict(sample)
            for t in range(len(label)):
                total_count += 1
                if label[t] == label_predict[t]:
                    correct_count += 1

        print('* testing finished')
        print('  correct: %d' % correct_count)
        print('  total: %d' % total_count)
        print('  performance: %f' % (correct_count / total_count))
        print('[%s] testing done' % datetime.datetime.now())

    def generate_observation(self, test_corpus_filename):
        test_data = self.read_file(test_corpus_filename)
        for sample, label in test_data:
            label_predict = self.predict(sample)
            for t in range(len(sample)):
                print('%s\t%s\t%s' % ('\t'.join(sample[t]), label[t], label_predict[t]))
            print()

    def predict(self, sample):
        """
        预测序列

        Parameters
        ----------
        sample
        """
        potential_table = self.generate_potential_table(self.parameter, self.label_number,
                                                        self.feature, sample, inference=True)
        label = self.viterbi(sample, potential_table)
        return label

    def viterbi(self, sample, potential_table):
        """
        维特比算法，预测随机序列

        Parameters
        ----------
        sample : list
            观测序列
        potential_table : list
        """
        sample = list(sample)
        potential_table = list(potential_table)

        time_length = len(sample)
        max_table = numpy.zeros((time_length, self.label_number))
        max_table_index = numpy.zeros((time_length, self.label_number), dtype='int64')

        t = 0
        for label_id in range(self.label_number):
            max_table[t, label_id] = potential_table[t][FIRST_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.label_number):
                max_value = -float('inf')
                max_label_id = None
                for previous_label_id in range(1, self.label_number):
                    value = max_table[t - 1, previous_label_id] * potential_table[t][previous_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = previous_label_id
                max_table[t, label_id] = max_value
                max_table_index[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length - 1].argmax()
        sequence.append(next_label)
        for t in range(time_length - 1, -1, -1):
            next_label = max_table_index[t, next_label]
            sequence.append(next_label)
        return [self.label_map[label_id] for label_id in sequence[::-1][1:]]

    def save(self, filename):
        """
        通过 json 保存模型

        Parameters
        ----------
        filename : str
            保存路径
        """
        model = {"feature_map": self.feature.encode_feature_map(),
                 "feature_number": self.feature.feature_number,
                 "label": self.feature.label_set,
                 "parameter": list(self.parameter)}
        f = open(filename, 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        import os
        print('* model saved')
        print('  at "%s/%s' % (os.getcwd(), filename))

    def load(self, path):
        """
        加载模型

        Parameters
        ----------
        path
        """
        file = open(path)
        model = json.load(file)
        file.close()

        self.feature = self.Feature()
        self.feature.load(model['feature_map'], model['feature_number'], model['label'])
        self.label_map, self.label = self.feature.get_label()
        self.label = list(self.label)
        self.label_number = len(self.label)
        self.parameter = numpy.asarray(model['parameter'])

        print('* load model')
        if self.debug_enable:
            print('  feature number: %d' % self.feature.feature_number)
            print('  label number: %d' % self.label_number)

    class Feature:

        # 标记 t = -1
        FIRST_LABEL = '*'
        # 标记索引
        FIRST_LABEL_INDEX = 0

        def __init__(self, feature_function=None):
            # 特征映射 特征字符串(由feature_function生成) -> 状态特征 / 过程特征 -> 特征标号
            self.feature_map = {}
            # 特征函数 为function类对象，调用执行生成特征字符
            self.feature_function = self.default_feature_function
            # 状态映射 状态字符串 -> 状态标号
            self.label_map = {self.FIRST_LABEL: self.FIRST_LABEL_INDEX}
            # 状态列表
            self.label = [self.FIRST_LABEL]
            # 特征数
            self.feature_number = 0
            # 特征经验统计模型 具有和特征映射相同的映射结构
            # 特征字符串(由feature_function生成) -> 状态特征 / 过程特征 -> 特征标号
            self.empirical_count = Counter()

            if feature_function is not None:
                self.feature_function = feature_function

        @staticmethod
        def default_feature_function(sample, t):
            """
            默认特征函数

            Parameters
            ----------
            sample :
                观测序列
            t : int
                时序

            Returns
            -------
            生成默认观测序列
            """
            length = len(sample)
            # 特征列表
            features = list()
            features.append('U[0]:%s' % sample[t][0])
            features.append('POS_U[0]:%s' % sample[t][1])
            # 对于序列长度 length - 1 以内的时序
            if t < length - 1:
                # 该时序下 后一字符为
                features.append('U[+1]:%s' % (sample[t + 1][0]))
                # 该时序下 当前字符为 同时后一字符为
                features.append('B[0]:%s %s' % (sample[t][0], sample[t + 1][0]))
                # 该时序下 后一字符词性为
                features.append('POS_U[+1]:%s' % sample[t + 1][1])
                # 该时序下 当前字符词性为 同时后一字符词性为
                features.append('POS_B[0]:%s %s' % (sample[t][1], sample[t + 1][1]))
                if t < length - 2:
                    # 该时序下 后二字符为
                    features.append('U[+2]:%s' % (sample[t + 2][0]))
                    # 该时序下 后二字符词性为
                    features.append('POS_U[+2]:%s' % (sample[t + 2][1]))
                    # 该时序下 后一字符词性为 同时后二字符词性为
                    features.append('POS_B[+1]:%s %s' % (sample[t + 1][1], sample[t + 2][1]))
                    # 该时序下 当前字符词性为 同时后一字符词性为 同时后二字符词性为
                    features.append('POS_T[0]:%s %s %s' % (sample[t][1], sample[t + 1][1], sample[t + 2][1]))
            if t > 0:
                # 该时序下 前一字符为
                features.append('U[-1]:%s' % (sample[t - 1][0]))
                # 该时序下 前一字符为 同时当前字符为
                features.append('B[-1]:%s %s' % (sample[t - 1][0], sample[t][0]))
                # 该时序下 前一字符词性为
                features.append('POS_U[-1]:%s' % (sample[t - 1][1]))
                # 该时序下 前一字符词性为 同时当前字符词性为
                features.append('POS_B[-1]:%s %s' % (sample[t - 1][1], sample[t][1]))
                if t < length - 1:
                    # 该时序下 前一字符词性为 同时当前字符词性为 同时后一字符词性为
                    features.append('POS_T[-1]:%s %s %s' % (sample[t - 1][1], sample[t][1], sample[t + 1][1]))
                if t > 1:
                    # 该时序下 前二字符为
                    features.append('U[-2]:%s' % (sample[t - 2][0]))
                    # 该时序下 前二字符词性为
                    features.append('POS_U[-2]:%s' % (sample[t - 2][1]))
                    # 该时序下 前二字符词性为 同时前一字符词性为
                    features.append('POS_B[-2]:%s %s' % (sample[t - 2][1], sample[t - 1][1]))
                    # 该时序下 前二字符词性为 同时前一字符词性为 同时当前字符此行为
                    features.append('POS_T[-2]:%s %s %s' % (sample[t - 2][1], sample[t - 1][1], sample[t][1]))

            return features

        def scan(self, data):
            """
            扫描观测序列，进行经验计数，生成特征模型。

            Parameters
            ----------
            data : list
                数据
            """
            # 遍历数据
            for i, j in data:
                previous_label = self.FIRST_LABEL_INDEX
                for t in range(len(i)):
                    # 这里生成状态标号 将特征字符形式转换为标号形式，便于计算，提高计算效率
                    # 如果不存在该状态则添加新状态
                    try:
                        label_id = self.label_map[j[t]]
                    except KeyError:
                        label_id = len(self.label_map)
                        self.label_map[j[t]] = label_id
                        self.label.append(j[t])
                    # 遍历 添加特征 这里提供 前一个标签 当前标签 观测序列 当前时序
                    self.add(previous_label, label_id, i, t)
                    previous_label = label_id

        def load(self, feature_map, feature_number, label):
            """
            加载特征模型

            Parameters
            ----------
            feature_map : dict
                特征映射
            feature_number : int
                特征数
            label :
            """
            self.feature_number = feature_number
            self.label = label
            self.label_map = {i: j for i, j in enumerate(label)}
            self.feature_map = self.decode_feature_map(feature_map)

        def __len__(self):
            return self.feature_number

        def add(self, previous_label, label, sample, t):
            """
            添加特征函数

            Parameters
            ----------
            previous_label : int
                前一状态标号
            label : int
                当前状态标号
            sample : list
                样本
            t : int
                时序
            """
            # 遍历特征函数
            for feature_string in self.feature_function(sample, t):
                # 如果特征字典中存在该特征
                if feature_string in self.feature_map.keys():
                    # 刷新转移特征
                    # 如果转移特征存在
                    if (previous_label, label) in self.feature_map[feature_string].keys():
                        # 经验计数加1
                        self.empirical_count[self.feature_map[feature_string][(previous_label, label)]] += 1
                    # 如果转移特征不存在 则创建转移特征
                    else:
                        feature_id = self.feature_number
                        self.feature_map[feature_string][(previous_label, label)] = feature_id
                        self.empirical_count[feature_id] += 1
                        self.feature_number += 1
                    # 刷新状态特征
                    # 如果状态特征存在
                    if (-1, label) in self.feature_map[feature_string].keys():
                        self.empirical_count[self.feature_map[feature_string][(-1, label)]] += 1
                    # 如果状态特征不存在 则创建状态特征
                    else:
                        feature_id = self.feature_number
                        self.feature_map[feature_string][(-1, label)] = feature_id
                        self.empirical_count[feature_id] += 1
                        self.feature_number += 1
                # 如果特征字典中并不存在该特征
                else:
                    self.feature_map[feature_string] = dict()
                    # 添加转移特征
                    feature_id = self.feature_number
                    self.feature_map[feature_string][(previous_label, label)] = feature_id
                    self.empirical_count[feature_id] += 1
                    self.feature_number += 1
                    # 添加状态特征
                    feature_id = self.feature_number
                    self.feature_map[feature_string][(-1, label)] = feature_id
                    self.empirical_count[feature_id] += 1
                    self.feature_number += 1

        def get_feature_vector(self, previous_label, label, sample, t):
            """
            获取特征向量

            Parameters
            ----------
            previous_label : int
                前一特征标号
            label : int
                当前特征标号
            sample : list
                观测序列
            t : int
                时序

            Returns
            -------
            list
            特征标号列表
            """
            features = list()
            # 遍历所有特征字符串
            for feature_string in self.feature_function(sample, t):
                try:
                    # 尝试从特征字典中搜索相关的特征函数标号 并添加
                    features.append(self.feature_map[feature_string][(previous_label, label)])
                except KeyError:
                    pass
            return features

        def get_label(self):
            """
            获取状态相关属性

            Returns
            -------
            label_map : dict
                状态映射
            label : list
                状态字符串列表
            """
            return self.label_map, self.label

        def calculate_conditional_probability(self, parameter, sample, t):
            """
            计算条件概率
            对特定序列在某一特定时序产生的所有特征字符串所对应的所有特征函数进行加和统计

            Parameters
            ----------
            parameter : list
                参数向量
            sample : list
                观测序列
            t : int
                时序

            Returns
            -------
            list [((previous_label, label), probability)]
            状态转移条件概率
            """
            parameter = list(parameter)
            conditional_probability = Counter()
            for feature_string in self.feature_function(sample, t):
                try:
                    for (previous_label, label), feature_id in self.feature_map[feature_string].items():
                        conditional_probability[(previous_label, label)] += parameter[feature_id]
                except KeyError:
                    pass
            return [((previous_label, label), probability)
                    for (previous_label, label), probability in conditional_probability.items()]

        def get_empirical_count(self):
            """
            获取经验统计模型

            Returns
            -------
            Counter
            """
            empirical_count = numpy.ndarray((self.feature_number,))
            for feature_id, counts in self.empirical_count.items():
                empirical_count[feature_id] = counts
            return empirical_count

        def get_feature_list(self, sample, t):
            """
            获取特征标号列表
            对特定序列在某一特定时序产生的所有特征字符串所对应的所有特征函数进行汇总

            Parameters
            ----------
            sample : list
                观测序列
            t : int
                时序

            Returns
            -------
            list
            获取列表
            """
            feature_list_map = {}
            for feature_string in self.feature_function(sample, t):
                for (previous_label, label), feature_id in self.feature_map[feature_string].items():
                    if (previous_label, label) in feature_list_map.keys():
                        feature_list_map[(previous_label, label)].add(feature_id)
                    else:
                        feature_list_map[(previous_label, label)] = {feature_id}
            return [((previous_label, label), feature_ids)
                    for (previous_label, label), feature_ids in feature_list_map.items()]

        def encode_feature_map(self):
            """
            生成特征模型的字符表现形式，用以静态存储

            Returns
            -------
            dict
            """
            encode = {}
            for feature_string in self.feature_map.keys():
                encode[feature_string] = {}
                for (previous_label, label), feature_id in self.feature_map[feature_string].items():
                    encode[feature_string]['%d_%d' % (previous_label, label)] = feature_id
            return encode

        @staticmethod
        def decode_feature_map(encode):
            """
            将特征模型的字符表现形式，转换为特征模型

            Parameters
            ----------
            encode : dict
                特征模型的字符表现形式
            """
            feature_map = {}
            for feature_string in encode.keys():
                feature_map[feature_string] = {}
                for transition_string, feature_id in encode[feature_string].items():
                    previous_label, label = transition_string.split('_')
                    feature_map[feature_string][(int(previous_label), int(label))] = feature_id
            return feature_map
