from matplotlib import pyplot

import leapy

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # load_iris()
    # -------------------------------------------------------------------------------------------------------------
    # 加载iris数据 {"handwritten_numeral":[[...] * 4](sepal length, sepal width, petal length, petal width), "target":[...]}
    # iris = load_iris()
    #
    # sample = numpy.mat([[x[1], x[3]] for x in iris.handwritten_numeral[0: 100]])
    # label = numpy.mat([[1 if y == 0 else -1 for y in iris.target[0: 100]]])
    #
    # vector_machine = support_vector_machine.SupportVectorMachine()
    # vector_machine.fit(sample, label)
    #
    # w = vector_machine.w
    # k = -w[0, 0] / w[1, 0]
    # b = vector_machine.b
    # x = numpy.linspace(min(sample.tolist(), key=lambda x: x[0])[0], max(sample.tolist(), key=lambda x: x[0])[0], 50)
    # y = (-w[0] / w[1] * x - b / w[1]).ravel()
    # a = vector_machine.a
    #
    # shift_b = pow((numpy.mat(w).T * numpy.mat(w))[0, 0], 1 / 2)
    # shift_b = abs(w[1, 0]) / shift_b
    # shift_b = [(1 / w[1, 0]) / shift_b] * len(x)
    #
    # y1 = (k * x - b / w[1, 0] + shift_b).ravel()
    # y2 = (k * x - b / w[1, 0] - shift_b).ravel()
    #
    # sample = numpy.mat(sample)
    # label = numpy.mat(label)
    #
    # pyplot.scatter(sample[:50, 0].T.tolist()[0], sample[:50, 1].T.tolist()[0], label='0', color="#ffb07c")
    # pyplot.scatter(sample[50:, 0].T.tolist()[0], sample[50:, 1].T.tolist()[0], label='1', color="#c94cbe")
    #
    # for i in range(len(a)):
    #     if a[i] > 0:
    #         pyplot.scatter(sample[i, 0], sample[i, 1], label="2", color="#087804")
    #
    # pyplot.plot(x.tolist(), y.tolist()[0], color="#087804")
    # pyplot.plot(x.tolist(), y1.tolist()[0], "--", color="#087804")
    # pyplot.plot(x.tolist(), y2.tolist()[0], "--", color="#087804")
    #
    # pyplot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease_origin.csv
    # ------------------------------------------------------------------------------------------------------------------
    sample_train, sample_test, label_train, label_test = leapy.load_heart_disease_data(split_enable=True)

    vector_machine = leapy.SupportVectorMachine()
    vector_machine.fit(sample_train, label_train)

    print(vector_machine.score(sample_test, label_test))

    # ------------------------------------------------------------------------------------------------------------------
    # linearly_indivisible.mat
    # ------------------------------------------------------------------------------------------------------------------
    sample_train, sample_test, label_train, label_test = \
        leapy.load_linearly_indivisible_data(split_enable=True)

    vector_machine = leapy.SupportVectorMachine()
    vector_machine.fit(sample_train, label_train, kernel="gaussian_kernel_function", p=[0.1])

    raw = len(sample_test)
    positive_x1 = []
    positive_x2 = []
    negative_x1 = []
    negative_x2 = []
    for i in range(raw):
        if label_test[i, 0] == 1:
            positive_x1.append(sample_test[i, 0])
            positive_x2.append(sample_test[i, 1])
        else:
            negative_x1.append(sample_test[i, 0])
            negative_x2.append(sample_test[i, 1])

    pyplot.scatter(positive_x1, positive_x2, label='0', color="#ffb07c")
    pyplot.scatter(negative_x1, negative_x2, label='1', color="#c94cbe")
    pyplot.show()

    label_predict = vector_machine.predict(sample_test)
    positive_x1 = []
    positive_x2 = []
    negative_x1 = []
    negative_x2 = []
    for i in range(raw):
        if label_predict[i, 0] == 1:
            positive_x1.append(sample_test[i, 0])
            positive_x2.append(sample_test[i, 1])
        else:
            negative_x1.append(sample_test[i, 0])
            negative_x2.append(sample_test[i, 1])

    pyplot.scatter(positive_x1, positive_x2, label='0', color="#ffb07c")
    pyplot.scatter(negative_x1, negative_x2, label='1', color="#c94cbe")
    pyplot.show()

    print(vector_machine.score(sample_test, label_test))
