from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

import numpy
import leapy


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease_data.csv
    # ------------------------------------------------------------------------------------------------------------------
    sample, label = leapy.load_heart_disease_data()
    sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.3)

    boosting = leapy.AdaptiveBoosting(debug_enable=True)
    boosting.fit(sample_train, label_train, learning_rate=0.5)
    boosting.score(sample_test, label_test)

    # ------------------------------------------------------------------------------------------------------------------
    # load_boston()
    # ------------------------------------------------------------------------------------------------------------------
    # boston = load_boston()
    # sample = numpy.mat(boston.data)
    # label = numpy.mat([boston.target]).T
    # model = leapy.BoostingRegressionTree(debug_enable=True)
    # model.fit(sample, label, dispersed=False, fit_threshold=100, trees_number=100)
    # model.save("models/boosting_regression_tree")
    # model.load("models/boosting_regression_tree")
    # label_predict = numpy.mat(model.predict(sample))
    # for i in range(sample.shape[0]):
    #     print(label_predict[i, 0], label[i, 0])
