from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

import leapy
import time


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease
    # ------------------------------------------------------------------------------------------------------------------
    # sample, label = leapy.load_heart_disease_data()
    #
    # k = 10
    # data_split = leapy.cross_validation(sample, label, k)
    #
    # score = []
    # perceptron = leapy.Perceptron(debug_enable=False)
    # for i in data_split:
    #     perceptron.fit(i[0], i[2])
    #     score.append(perceptron.score(i[1], i[3]))
    # score = sum(score) / k
    #
    # print('---------------------------')
    # print('score: ' + str(score))
    # print('---------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # sklearn heart_disease
    # ------------------------------------------------------------------------------------------------------------------
    sample_train, sample_test, label_train, label_test = leapy.load_heart_disease_data(split_enable=True)

    perceptron = leapy.Perceptron()

    start = time.time()
    perceptron.fit(sample_train, label_train)
    end = time.time()
    leapy_time = end - start
    leapy_score = perceptron.score(sample_test, label_test)

    perceptron = Perceptron()
    start = time.time()
    perceptron.fit(sample_train, label_train)
    end = time.time()
    sklearn_time = end - start
    sklearn_score = perceptron.score(sample_test, label_test)

    print()
    print('-----------------------------')
    print("time: ")
    print("  leapy:   " + str(leapy_time))
    print("  sklearn: " + str(sklearn_time))
    print('-----------------------------')
    print("score: ")
    print("  leapy:   " + str(leapy_score))
    print("  sklearn: " + str(sklearn_score))
