from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from matplotlib import pyplot

import numpy
import leapy


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------------------------------------------------------
    # sample, label = leapy.load_loan_data()
    # decision_tree = leapy.IterativeDichotomiser3Tree(debug_enable=True)
    # decision_tree.fit(sample, label)
    # decision_tree.score(sample, label)

    # ------------------------------------------------------------------------------------------------------------------
    # sklearn.datasets.load_boston()
    # ------------------------------------------------------------------------------------------------------------------
    # handwritten_numeral = load_boston()
    #
    # sample = handwritten_numeral["handwritten_numeral"]
    # label = handwritten_numeral["target"]
    #
    # sample_train, sample_test, label_train, label_test = \
    #     train_test_split(sample, label, test_size=0.3)
    #
    # sample_train = numpy.mat(sample_train)
    # label_train = numpy.mat([label_train]).T
    #
    # sample_test = numpy.mat(sample_test)
    # label_test = numpy.mat(label_test).T
    #
    # model = leapy.RegressionTree(debug_enable=True)
    # model.fit(sample_train, label_train, max_deep=100)
    #
    # print(model.score(sample_test, label_test))
    #
    # label_predict = model.predict(sample_test)
    #
    # color_map = pyplot.get_cmap('viridis')
    # index = sample_train[:, 0].argsort(0)
    # pyplot.scatter(sample_train[index, 0].T.tolist()[0], label_train[index, 0].T.tolist()[0],
    #                color=color_map(0.9), s=10)
    # index = sample_test[:, 0].argsort(0)
    # pyplot.scatter(sample_test[index, 0].T.tolist()[0], label_test[index, 0].T.tolist()[0],
    #                color=color_map(0.5), s=10)
    #
    # pyplot.plot(sample_test[index, 0].T.tolist()[0], label_predict[index, 0].T.tolist()[0], color="black")
    # pyplot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease_origin.csv
    # ------------------------------------------------------------------------------------------------------------------
    # sample, label = leapy.load_heart_disease_data()
    #
    # sample_positive = []
    # sample_negative = []
    # label_positive = []
    # label_negative = []
    #
    # # 分类正反例
    # for i, j in zip(sample, label):
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
    # sample_test = sample_positive_test + sample_negative_test
    # label_test = label_positive_test + label_negative_test
    #
    # decision_tree = leapy.ClassificationTree()
    # decision_tree.fit(sample_train, label_train, dispersed=[True, False, False, True, True, False,
    #                                                         False, True, False, True, False, False, False])
    #
    # test_number = len(sample_test)
    # right_number = 0
    # for i in range(len(sample_test)):
    #     if decision_tree.predict(sample_test[i]) == label_test[i]:
    #         right_number += 1
    # print(right_number / test_number)

    # ------------------------------------------------------------------------------------------------------------------
    # credit_card_data.csv
    # ------------------------------------------------------------------------------------------------------------------
    sample_train, sample_test, label_train, label_test = leapy.load_credit_card_data(split_enable=True)

    model = leapy.IterativeDichotomiser3Tree(debug_enable=True)
    model.fit(sample_train, label_train)
    model.save("models/iterative_dichotomiser_3_tree")
    model.load("models/iterative_dichotomiser_3_tree")
    model.score(sample_test, label_test)
