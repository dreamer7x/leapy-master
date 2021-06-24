from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import numpy
import leapy
import time


if __name__ == '__main__':
    sample_train, sample_test, label_train, label_test = leapy.load_heart_disease_data(split_enable=True)

    model = leapy.NaiveBayesianClassifier()
    start = time.time()
    model.fit(sample_train, label_train)
    end = time.time()
    leapy_time = end - start
    leapy_score = model.score(sample_test, label_test)

    # sklearn 朴素贝叶斯模块
    model = GaussianNB()
    start = time.time()
    model.fit(sample_train, numpy.array(label_train).ravel())
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
