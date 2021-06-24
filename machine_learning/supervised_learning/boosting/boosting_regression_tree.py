import numpy
import leapy


class BoostingRegressionTree:
    def __init__(self, debug_enable=False):
        """
        初始化提升回归树

        Parameters
        ----------
        debug_enable : bool
            是否调试 True将进行过程反馈
        """
        # 调试
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        # 回归树数量上限
        self.max_trees_number = 0
        # 学习率
        self.learning_rate = 0
        # 拟合阈值
        self.fit_threshold = 0
        # 树深度
        self.max_deep = 0
        # 弱分类器集合
        self.trees = None
        # 训练样本
        self.sample = None
        # 训练标签
        self.label = None
        # 离散化信息
        self.dispersed = None
        # 残差列表
        self.r = None

    def fit(self, sample, label,
            learning_rate=1.0, fit_threshold=100,
            dispersed=None, trees_number=20, max_deep=2):
        """

        Parameters
        ----------
        sample : numpy.array N * p, numpy.matrix N * p, list N * [p]
            样本
        label : numpy.array N * 1 / 1 * N, numpy.matrix N * 1 / 1 * N, list N
            标签
        learning_rate : float
            学习率
        fit_threshold : float
            拟合阀值
        dispersed : list, bool
            离散化信息
        trees_number : int
            回归树数量上限
        max_deep : int
            回归树深度上限
        """
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
        self.label = label
        self.r = self.label.copy()
        # 回归树列表
        self.trees = []
        if dispersed is None or dispersed is False:
            dispersed = [False] * col
        if dispersed is True:
            dispersed = [True] * col
        self.dispersed = dispersed
        self.max_trees_number = trees_number
        self.learning_rate = learning_rate
        self.fit_threshold = fit_threshold
        self.max_deep = max_deep

        while True:
            l = self.get_loss()
            if l <= self.fit_threshold:
                if self.debug_enable:
                    print("报告: BoostingRegressionTree fit 拟合完成 拟合达到理想状态")
                    print("     trees_number: " + str(len(self.trees)))
                    print("     loss: " + str(self.get_loss()))
                break

            if self.debug_enable:
                print("报告: BoostingRegressionTree fit 损失值: " + str(l))
            tree = self.get_tree()
            self.update_r(tree)
            self.trees.append(tree)

            if len(self.trees) >= self.max_trees_number:
                if self.debug_enable:
                    print("报告: BoostingRegressionTree fit 拟合完成")
                    print("     trees_number: " + str(len(self.trees)))
                    print("     loss: " + str(self.get_loss()))
                break

    # 构造弱选择器模型
    def get_tree(self):
        tree = leapy.RegressionTree(debug_enable=self.debug_enable)
        tree.fit(self.sample, self.r, dispersed=self.dispersed, max_deep=self.max_deep)
        return tree

    def update_r(self, tree):
        """
        更新残差值

        Parameters
        ----------
        tree : RegressionTree
            弱选择器
        """
        raw, col = self.sample.shape
        for i in range(raw):
            label_predict = tree.forward(self.sample[i, :], )
            self.r[i, 0] -= label_predict

    def get_loss(self):
        """
        计算残差方差，表示残差的缩小范围，注意这里计算的并不是预测误差方差

        Returns
        -------
        float 残差方差
        """
        raw, col = self.sample.shape
        average = sum(self.r.T.tolist()[0]) / raw
        variance = 0
        for i in range(raw):
            variance += (self.r[i, 0] - average) ** 2
        return variance

    def predict(self, sample):
        """
        回归

        Parameters
        ----------
        sample : numpy.matrix N * p
            样本

        Returns
        -------
        numpy.matrix N * 1
        预测标签
        """
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(numpy.zeros((raw, 1)))
        for i in range(raw):
            label_predict = 0
            for j in range(len(self.trees)):
                label_predict += self.trees[j].forward(sample[i, :], )
            label[i, 0] = label_predict
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

        label_predict = numpy.mat(self.predict(sample))
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
        self.trees = []
        for i in encode:
            tree = leapy.RegressionTree(debug_enable=self.debug_enable)
            tree.decode(i)
            self.trees.append(tree)

    def encode(self):
        encode = []
        for i in self.trees:
            encode.append(i.encode())
        return encode
