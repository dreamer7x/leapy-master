import numpy
import datetime


class Perceptron:
    def __init__(self,
                 debug_enable=False,
                 learning_rate=0.1,
                 iteration=1000,
                 algorithm=None):
        """
        Parameters
        ----------
        debug_enable : bool
            是否调试
        learning_rate : float
            学习率
        iteration : int
            迭代次数
        algorithm : str
            算法名
        """
        # 调试
        self.debug_enable = debug_enable

        # 初始化拟合过程参数 以下参数均在调用拟合方法时进行设置
        # 参数
        self.w = None
        # 超平面截距
        self.b = None
        # 学习率
        self.learning_rate = learning_rate
        # 迭代上限
        self.iteration = iteration
        # 优化算法
        if algorithm is None:
            self.algorithm = "batch_gradient_descent"
        else:
            self.algorithm = algorithm

    def batch_gradient_descent(self, sample, label):
        print("* Start batch_gradient_descent BGD")
        if self.debug_enable:
            print('  learning rate: ' + str(self.learning_rate))
            print('================================================')
            print('    iteration : wrong_count                     ')
            print('------------------------------------------------')

        raw, col = sample.shape
        w = numpy.mat(numpy.ones((col, 1)))
        b = 0

        iteration = 0
        while True:
            iteration += 1
            wrong_count = 0
            # 遍历样本点
            for i in range(raw):
                # 如果发现误分类点，就对参数进行一次修改
                if label[i, 0] * (sample[i, :] * w + b)[0, 0] <= 0:
                    # 梯度下降法
                    w += self.learning_rate * (label[i, 0] * sample[i, :]).T
                    b += self.learning_rate * label[i, 0]
                    # 计算一次遍历的误分类点数量
                    wrong_count += 1

            if self.debug_enable:
                print('   ', '{0:03d}'.format(iteration), ':', wrong_count)

            # 如果误分类数量为0，即所有点都得到正确分类
            if wrong_count == 0:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished linearly_indivisible')
                if self.debug_enable:
                    print('  iteration: ' + str(iteration))
                    print('  w: \n\n' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]), end='\n\n')
                    print('  b: ' + str(b))
                break

            if iteration >= self.iteration:
                if self.debug_enable:
                    print('================================================')
                print('* Training finished reach iteration')
                if self.debug_enable:
                    print('  w: \n\n' + " ".join([str(i) for i in w[:, 0].T.tolist()[0]]), end='\n\n')
                    print('  b: ' + str(b))
                print('  score: ' + str((raw - wrong_count) / raw))
                break

        return w, b

    # 随机梯度下降法
    def fit(self, sample, label):
        # --------------------------------------------------------------------------------------------------------------
        # 初始化
        # --------------------------------------------------------------------------------------------------------------
        sample = numpy.mat(sample)
        raw, col = sample.shape

        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print('------------------------------------------------')
            print('* Error from Classification.fit()             \n'
                  '  The number of samples and tags are different  ')
            print('------------------------------------------------')
            return

        # --------------------------------------------------------------------------------------------------------------
        # 拟合
        # --------------------------------------------------------------------------------------------------------------
        print('[%s] Start training' % datetime.datetime.now())
        if self.debug_enable:
            print("  number of sample: %d" % raw)

        if self.algorithm == "batch_gradient_descent":
            self.w, self.b = self.batch_gradient_descent(sample, label)

        print('[%s] Training done' % datetime.datetime.now())

    def predict(self, sample):
        sample = numpy.mat(sample)
        label = sample * self.w + self.b
        label = numpy.mat([[1 if i > 0 else -1 for i in label[:, 0]]]).T
        return label

    def score(self, sample, label):
        label_predict = self.predict(sample)
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label_predict.shape[0] != label.shape[0]:
            print('------------------------------------------------')
            print('* Error from Classification.fit()             \n'
                  '  The number of samples and tags are different  ')
            print('------------------------------------------------')
            return
        true_number = 0
        for j in range(label_predict.shape[0]):
            if label_predict[j] == label[j]:
                true_number += 1
        return true_number / len(label_predict)
