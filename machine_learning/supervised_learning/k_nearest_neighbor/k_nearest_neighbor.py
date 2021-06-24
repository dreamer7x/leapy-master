from collections import Counter
from collections import namedtuple

import numpy
import math
import datetime


class KNearestNeighbor:

    def __init__(self, debug_enable=False, k=1, p=2, algorithm=None):
        """
        Parameters
        ----------
        debug_enable : bool
            Whether to enter the debugging state, enter the feedback process information.
            是否进入调试状态，进入将反馈过程信息

        k : int
            Number of nearest_neighbors 最近邻数量

        p : int
            Distance measurement formula 距离度量公式
            1. Manhattan distance 曼哈顿距离
                math:
                    \sum_{i = 1}^n|x_i - y_i|
            2. Euclidean distance 欧氏距离
                math:
                    (\sum_{i = 1}^n|x_i - y_i|^2)^{\frac{1}{2}}
            3. Minkowski Distance 闵氏距离
                math:
                    (\sum_{i = 1}^n|x_i - y_i|^\infty)^{\frac{1}{\infty}}

        algorithm :
            To specify the algorithm, which supports:
            1. Normal_Solving solves the problem according to the concept of
               K-nearest neighbor model and minimizes the distance of all samples.
            2. K dimensional tree is used to construct the tree structure
               to improve the efficiency of problem solving.

            用以指定算法，支持：
            1. 概念求解 (normal_solving) 按照 K近邻 模型的概念进行求解，对所有样本求最小距离
            2. K维分类树 (K_dimensional_tree) 构造树结构，从而提高问题求解效率
        """
        # 是否调试
        self.debug_enable = debug_enable
        self.k = k
        self.p = p
        # 算法
        if algorithm is None:
            self.algorithm = "k_dimensional_tree"
        else:
            self.algorithm = algorithm
        # 生成模型
        self.model = None

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
        print('* load_data ... ')
        if self.debug_enable:
            print("  number of sample: %d" % raw)

        if self.algorithm == "normal_solving":
            self.model = {"sample_train": sample, "label_train": label}
        elif self.algorithm == "k_dimensional_tree" or self.algorithm == "KDtree":
            self.model = {"root": self.create_tree(0, sample, label)}
        else:
            self.model = {"root": self.create_tree(0, sample, label)}

        print('[%s] Training done' % datetime.datetime.now())

    # 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
    # 轻量级字典 用以存储少量数据
    Result = namedtuple("result", "nearest_coordinate nearest_distance visited_number value")

    class KDimensionalTreeNode:
        def __init__(self, dimension, coordinate, value, left, right):
            # 分割维度
            self.dimension = dimension
            # 分割坐标 即样本
            self.coordinate = coordinate
            # 分割目标值 即标签
            self.value = value
            # 左子节点
            self.left = left
            # 右子节点
            self.right = right

    # 计算点距离
    @staticmethod
    def calculate_distance(x1, x2, p):
        x1 = numpy.mat(x1)
        x2 = numpy.mat(x2)
        return math.pow(numpy.sum(numpy.power(numpy.abs(x1 - x2), float(p))), 1 / p)

    def create_tree(self, dimension, coordinates, values):
        # 数据集为空 则跳出递归
        if coordinates.shape[0] == 0:
            return
        raw, col = coordinates.shape
        # 排序
        index = coordinates[:, dimension].argsort(axis=0)
        coordinates = coordinates[index][:, 0, :]
        values = values[index]
        # 整除
        split_index = raw // 2
        # 中位数分割点
        split_coordinate = coordinates[split_index, :]
        split_value = values[split_index, 0][0, 0]
        # 遍历切割
        dimension_next = (dimension + 1) % col
        # 递归的创建kd树
        return self.KDimensionalTreeNode(dimension,
                                         split_coordinate,
                                         split_value,
                                         # 递归创建左子树
                                         self.create_tree(dimension_next, coordinates[:split_index, :],
                                                          values[:split_index, 0]),
                                         # 递归创建右子树
                                         self.create_tree(dimension_next, coordinates[split_index + 1:, :],
                                                          values[split_index + 1:, 0]))  # 递归创建右子树

    def travel(self, node, target, max_distance):
        raw, col = target.shape
        # 如果子节点为空 则说明已经到达叶节点 跳出递归
        if node is None:
            return self.Result([[0] * col] * self.k, [float("inf")] * self.k, 0, [None] * self.k)
        # 初始化跳转数量
        visited_number = 1
        # 分割维度
        split_dimension = node.dimension
        # 分割坐标
        split_coordinate = node.coordinate
        # 分割标签
        split_value = node.value

        # 目标点第k维数值小于分割轴第k维数值
        if target[0, split_dimension] <= split_coordinate[0, split_dimension]:
            nearer_node = node.left
            further_node = node.right
        # 目标点第k维数值大于分割轴第k维数值
        else:
            nearer_node = node.right
            further_node = node.left

        # 进行遍历找到包含目标点的区域
        temporary_result1 = self.travel(nearer_node, target, max_distance)

        # 当前最近邻点的最远点
        nearest = temporary_result1.nearest_coordinate
        # 当前最近邻点的最远距离
        distance = temporary_result1.nearest_distance
        # 叠加跳转次数
        visited_number += temporary_result1.visited_number
        # 更新最近标签
        value = temporary_result1.value

        # 最近点将在以目标点为球心，max_dist为半径的超球体内
        if distance[-1] < max_distance:
            max_distance = distance[-1]

        # 第k维上目标点与分割超平面的距离
        temporary_distance = abs(split_coordinate[0, split_dimension] - target[0, split_dimension])
        # 判断超球体是否与超平面相交
        if max_distance < temporary_distance:
            # 不相交 则跳出递归
            return temporary_result1

        # 计算目标点与分割点距离
        temporary_distance = self.calculate_distance(split_coordinate, target, p=self.p)

        # 如果存在更近距离
        if temporary_distance < distance[-1]:
            # 更新最近点
            nearest[-1] = split_coordinate
            # 更新最近距离
            distance[-1] = temporary_distance
            # 更新超球体半径
            max_distance = distance[-1]
            # 更新标签
            value[-1] = split_value
            # 排序
            index = [i[0] for i in sorted(enumerate(distance), key=lambda x: x[1])]
            nearest = [nearest[i] for i in index]
            distance = [distance[i] for i in index]
            value = [value[i] for i in index]

        # 检查另一个子结点对应的区域是否有更近的点
        temporary_result2 = self.travel(further_node, target, max_distance)

        visited_number += temporary_result2.visited_number

        for i in range(self.k):
            # 如果存在更近距离
            if temporary_result2.nearest_distance[i] < distance[-1]:
                nearest[-1] = temporary_result2.nearest_coordinate[i]  # 更新最近点
                distance[-1] = temporary_result2.nearest_distance[i]  # 更新最近距离
                value[-1] = temporary_result2.value[i]
                # 排序
                index = [i[0] for i in sorted(enumerate(distance), key=lambda x: x[1])]
                nearest = [nearest[i] for i in index]
                distance = [distance[i] for i in index]
                value = [value[i] for i in index]

        return self.Result(nearest, distance, visited_number, value)

    def find_nearest_node(self, point):
        point = numpy.mat(point)
        return self.travel(self.model["root"], point, float("inf"))

    def normal_solving(self, target):
        target = numpy.mat(target)

        k_nearest_neighbors = [(float("inf"), None, None)] * self.k
        # 获取最近临点
        max_index = k_nearest_neighbors.index(max(k_nearest_neighbors, key=lambda n: n[0]))
        for j in range(self.model["sample_train"].shape[0]):
            distance = self.calculate_distance(target, self.model["sample_train"][j, :], p=self.p)
            if k_nearest_neighbors[max_index][0] > distance:
                k_nearest_neighbors[max_index] = (distance,
                                                  self.model["sample_train"][j, :],
                                                  self.model["label_train"][j, 0])
                max_index = k_nearest_neighbors.index(max(k_nearest_neighbors, key=lambda n: n[0]))
        return k_nearest_neighbors

    def predict(self, sample):
        sample = numpy.mat(sample)
        raw, col = sample.shape

        if self.algorithm is None or self.algorithm == "math_solving":
            label = numpy.mat(numpy.empty(raw, 1))
            for i in range(raw):
                k_nearest_neighbors = self.normal_solving(sample[i, :])
                # 统计
                counter = Counter([j[-1] for j in k_nearest_neighbors])
                max_count = sorted(counter, key=lambda c: c)[-1]
                label[i, 0] = max_count
            return label

        if self.algorithm == "k_dimensional_tree" or self.algorithm == "KDtree":
            label = numpy.mat(numpy.empty((raw, 1)))
            for i in range(raw):
                result = self.find_nearest_node(sample[i, :])
                counter = Counter([i for i in result.value])
                label[i, 0] = sorted(counter, key=lambda c: c)[-1]
            return label

    def score(self, sample, label):
        sample = numpy.mat(sample)
        raw, col = sample.shape
        label = numpy.mat(label)
        if label.shape[0] == 1:
            label = label.T
        if label.shape[0] != raw:
            print('--------------------------------------------------------')
            print('* Error from Classification.fit()                     \n'
                  '  The number of samples and tags are different          ')
            print('--------------------------------------------------------')
            return

        label_predict = self.predict(sample)
        right_count = 0
        for i in range(label_predict.shape[0]):
            if label[i] == label_predict[i]:
                right_count += 1
        return right_count / sample.shape[0]

    def load(self, file_path):
        file = open(file_path, "r")
        data = file.read()
        information, encode = data.split("\n", maxsplit=1)

        information = eval(information)
        if information["algorithm"] == "k_dimensional_tree":
            self.model["root"] = self.k_dimension_tree_decode(encode)
            self.algorithm = "k_dimensional_tree"
        else:
            print('--------------------------------------------------------')
            print('* Error from Classification.load()                    \n'
                  '  Wrong model information, the algorithm does not exist.')
            print('--------------------------------------------------------')
            return

        file.close()

    def save(self, file_path):
        file = open(file_path, "w")

        if self.algorithm == "k_dimensional_tree":
            encode = self.k_dimension_tree_encode(self.model["root"])
            information = {"algorithm": "k_dimensional_tree"}
            encode = str(information) + "\n" + encode
            file.write(encode)

        file.close()

    def k_dimension_tree_decode(self, encode):
        def to_tree(d):
            if d is None:
                return None
            node = self.KDimensionalTreeNode(
                dimension=d["dimension"],
                coordinate=numpy.mat([d["coordinate"]]),
                value=d["value"],
                left=to_tree(d["left"]),
                right=to_tree(d["right"])
            )
            return node
        dictionary = eval(encode)
        return to_tree(dictionary)

    @staticmethod
    def k_dimension_tree_encode(node):
        def to_dict(n):
            if n is None:
                return None
            dictionary = {
                "dimension": n.dimension,
                "coordinate": n.coordinate.tolist()[0],
                "value": n.value,
                "left": to_dict(n.left),
                "right": to_dict(n.right)
            }
            return dictionary
        return str(to_dict(node))
