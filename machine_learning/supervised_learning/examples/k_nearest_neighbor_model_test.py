from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from matplotlib import pyplot

import numpy
import pandas
import leapy
import time

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # load_iris()
    # -------------------------------------------------------------------------------------------------------------
    # # 取实验数据 为字典类型
    # # {"handwritten_numeral":[[...], [...], ...], "target":[...]}
    # iris = load_iris()
    # # 将其注入DataFrame类型
    # # 参数1: 样本数据(矩阵形式)
    # # 参数2: 属性名(属性名字符串列表)
    # data_frame = pandas.DataFrame(iris.data, columns=iris.feature_names)
    # data_frame['label'] = iris.target
    # # 覆写属性名 覆写数量不匹配将报错
    # data_frame.columns = ['x1', 'x2', 'x3', 'x4', 'label']
    #
    # # 读取
    # data = numpy.array(data_frame.iloc[:100, [0, 1, -1]])
    # sample, label = data[:, :-1], data[:, -1]
    # sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.3)
    #
    # model = leapy.KNearestNeighbor(algorithm="k_dimensional_tree", k=5)
    # model.fit(sample_train, label_train)
    # print(model.score(sample_test, label_test))
    #
    # test_point = [6.0, 3.0]
    # nearest_point = model.find_nearest_node(test_point)
    #
    # color_map = pyplot.get_cmap('viridis')
    # pyplot.scatter(data_frame[:50]['x1'],
    #                data_frame[:50]['x2'],
    #                color=color_map(0.7), s=10,
    #                label='0')
    #
    # pyplot.scatter(data_frame[50:100]['x1'],
    #                data_frame[50:100]['x2'],
    #                color=color_map(0.4), s=10,
    #                label='1')
    #
    # pyplot.scatter(test_point[0],
    #                test_point[1],
    #                color=color_map(0.9), s=10,
    #                label='test_point')
    #
    # pyplot.scatter([i[0, 0] for i in nearest_point.nearest_coordinate],
    #                [i[0, 1] for i in nearest_point.nearest_coordinate],
    #                color=color_map(0), s=10,
    #                label='nearest_point')
    #
    # pyplot.xlabel('x1')
    # pyplot.ylabel('x2')
    # pyplot.legend()
    # pyplot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease
    # -------------------------------------------------------------------------------------------------------------
    # sample, label = leapy.load_heart_disease_data()
    #
    # sample_positive = []
    # sample_negative = []
    # label_positive = []
    # label_negative = []
    #
    # # 分类正反例
    # for i, j in zip(sample.tolist(), label.T.tolist()[0]):
    #     if j == 1:
    #         sample_positive.append(i)
    #         label_positive.append(j)
    #     else:
    #         sample_negative.append(i)
    #         label_negative.append(j)
    #
    # sample_positive_train, sample_positive_test, label_positive_train, label_positive_test = \
    #     train_test_split(sample_positive, label_positive, test_size=0.3)
    # sample_negative_train, sample_negative_test, label_negative_train, label_negative_test = \
    #     train_test_split(sample_negative, label_negative, test_size=0.3)
    #
    # sample_train = numpy.array(sample_negative_train + sample_positive_train)
    # label_train = numpy.array(label_negative_train + label_positive_train)
    #
    # index = numpy.arange(len(sample_train))
    # numpy.random.shuffle(index)
    #
    # sample_train = sample_train[index]
    # label_train = label_train[index]
    #
    # model = leapy.KNearestNeighbor(algorithm="k_dimensional_tree")
    # model.load("models/k_dimensional_tree")
    # # model.fit(sample_train, label_train)
    # # model.save("models/k_dimensional_tree")
    #
    # sample_test = sample_positive_test + sample_negative_test
    # label_test = label_positive_test + label_negative_test
    #
    # print(model.score(sample_test, label_test))

    # ------------------------------------------------------------------------------------------------------------------
    # sklearn heart_disease
    # -------------------------------------------------------------------------------------------------------------
    sample, label = leapy.load_heart_disease_data()
    sample_positive = []
    sample_negative = []
    label_positive = []
    label_negative = []

    # 分类正反例
    for i, j in zip(sample.tolist(), label.T.tolist()[0]):
        if j == 1:
            sample_positive.append(i)
            label_positive.append(j)
        else:
            sample_negative.append(i)
            label_negative.append(j)

    sample_positive_train, sample_positive_test, label_positive_train, label_positive_test = \
        train_test_split(sample_positive, label_positive, test_size=0.3)
    sample_negative_train, sample_negative_test, label_negative_train, label_negative_test = \
        train_test_split(sample_negative, label_negative, test_size=0.3)

    sample_train = numpy.array(sample_negative_train + sample_positive_train)
    label_train = numpy.array(label_negative_train + label_positive_train)

    sample_test = sample_positive_test + sample_negative_test
    label_test = label_positive_test + label_negative_test

    model = KNeighborsClassifier(n_neighbors=3)

    start = time.time()
    model.fit(sample_train, label_train)
    end = time.time()
    leapy_time = end - start
    leapy_score = model.score(sample_test, label_test)

    model = leapy.KNearestNeighbor(k=3)

    start = time.time()
    model.fit(sample_train, label_train)
    end = time.time()
    sklearn_time = end - start
    sklearn_score = model.score(sample_test, label_test)

    print()
    print('-----------------------------')
    print("time: ")
    print("  leapy:   " + str(leapy_time))
    print("  sklearn: " + str(sklearn_time))
    print('-----------------------------')
    print("score: ")
    print("  leapy:   " + str(leapy_score))
    print("  sklearn: " + str(sklearn_score))

