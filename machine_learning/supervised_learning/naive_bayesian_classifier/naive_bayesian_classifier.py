import math
import pandas
import numpy
import datetime


class NaiveBayesianClassifier:
    def __init__(self, debug_enable=False):
        self.debug_enable = debug_enable
        # 朴素贝叶斯模型
        # 条件概率模型
        self.conditional_probability = None
        # 先验概率模型
        self.prior_probability = None

    @staticmethod
    def calculate_average(x):
        """
        Calculate the standard formula for the mean value.
        The numpy.mean() method may lose some precision.
        求平均值标准公式，使用 numpy.mean() 方法可能损失一定的精确度

        Parameters
        ----------
        x : numpy.matrix shape of N * p

        Returns
        -------
        numpy.matrix shape of 1 * p
        """
        return sum(x) / len(x)

    @staticmethod
    def calculate_standard_deviation(x, average):
        """
        To calculate the standard deviation formula.
        The numpy.var() method may lose some precision
        求标准差公式，使用 numpy.var() 方法可能损失一定的精确度

        Parameters
        ----------
        x : numpy.matrix shape of N * p
        average : numpy.matrix shape of 1 * p

        Returns
        -------
        float
        """
        return math.sqrt(sum([pow(x - average, 2) for x in x]) / float(len(x)))

    @staticmethod
    def calculate_gaussian_probability(x, average, variance):
        # 正态分布公式
        return 1 / (math.sqrt(2 * math.pi) * variance) * \
               math.exp(-(math.pow(x - average, 2) / (2 * math.pow(variance, 2))))

    def generate_conditional_probability(self, x):
        """
        Generating normal distribution model 生成正态分布模型

        Parameters
        ----------
        x : object numpy.matrix shape of N * p

        Returns
        -------
        summaries : list
        """
        conditional_probability = []
        for i in zip(*x):
            average = self.calculate_average(i)
            conditional_probability.append((average, self.calculate_standard_deviation(i, average)))
        return conditional_probability

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

        # 统计标签
        label_count = pandas.value_counts(label.T.tolist()[0])
        # 初始化朴素贝叶斯模型: {标签: [(样本均值， 样本方差(正态模型参数))]}
        conditional_probability = {i: [] for i in label_count.keys()}
        # 生成朴素贝叶斯模型
        for i, j in zip(sample.tolist(), label.T.tolist()[0]):
            conditional_probability[j].append(i)
        self.conditional_probability = {i: self.generate_conditional_probability(j)
                                        for i, j in conditional_probability.items()}
        self.prior_probability = {i: label_count[i] / raw for i in label_count.keys()}

        print('[%s] Training done' % datetime.datetime.now())

    def calculate_posterior_probability(self, sample):
        """
        Calculate of posterior probability 计算后验概率

        Parameters
        ----------
        sample : numpy.matrix shape of N * p

        Returns
        -------
        probability : dict
        """
        # model:    {0.0: [(5.0, 0.37), (3.42, 0.40)], 1.0: [(5.8, 0.449), (2.7, 0.27)]}
        # sample:   [1.1, 2.2]
        probability = {}
        for label, conditional_probability in self.conditional_probability.items():
            probability[label] = self.prior_probability[label]
            for i in range(len(conditional_probability)):
                average, variance = conditional_probability[i]
                probability[label] *= self.calculate_gaussian_probability(sample[i], average, variance)
        return probability

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(numpy.empty((raw, 1)))
        for i in range(raw):
            label[i, 0] = sorted(self.calculate_posterior_probability(sample[i, :].tolist()[0]).items(),
                                 key=lambda x: x[-1])[-1][0]
        return label

    def score(self, sample, label):
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

        label_predict = self.predict(sample)
        right_count = 0
        for i in range(raw):
            if label[i, 0] == label_predict[i, 0]:
                right_count += 1
        return right_count / len(sample)
