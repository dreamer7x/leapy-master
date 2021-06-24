from matplotlib import pyplot

import leapy

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # linearity_data.csv
    # ------------------------------------------------------------------------------------------------------------------
    # sample_train, sample_test, label_train, label_test = leapy.load_linearity_data(split_enable=True)
    #
    # # math_solving
    # # batch_gradient_descent
    # # mini_batch_gradient_descent
    # # stochastic_gradient_descent
    # linear_regression = leapy.StandardLinearRegression(debug_enable=True,
    #                                                    algorithm="math_solving",
    #                                                    threshold_enable=True)
    # # linear_regression = leapy.RidgeLinearRegression(debug_enable=True,
    # #                                                 algorithm="stochastic_gradient_descent",
    # #                                                 threshold_enable=True)
    # # linear_regression = leapy.LassoLinearRegression(debug_enable=True,
    # #                                                 algorithm="batch_gradient_descent",
    # #                                                 threshold_enable=True)
    # # linear_regression = leapy.LocalLinearRegression(debug_enable=True)
    #
    # linear_regression.fit(sample_train, label_train)
    # label_predict = numpy.array(linear_regression.predict(sample_test))
    #
    # color_map = pyplot.get_cmap('viridis')
    # pyplot.scatter(sample_train.T.tolist()[0], label_train.T.tolist()[0], color=color_map(0.9), s=10)
    # pyplot.scatter(sample_test.T.tolist()[0], label_test.T.tolist()[0], color=color_map(0.5), s=10)
    # pyplot.plot(sample_test.T.tolist()[0], label_predict.T.tolist()[0], color="black")
    # pyplot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # non_linearity_data.txt
    # ------------------------------------------------------------------------------------------------------------------
    # sample_train, sample_test, label_train, label_test = leapy.load_sine_data(split_enable=True)
    #
    # linear_regression = leapy.StandardLinearRegression(debug_enable=True, algorithm="match_solving",
    #                                                    polynomial_degree=10)
    # # linear_regression = leapy.RidgeLinearRegression(debug_enable=True, algorithm="math_solving",
    # #                                                 polynomial_degree=40)
    # # linear_regression = leapy.LassoLinearRegression(debug_enable=True, algorithm="math_solving")
    # # linear_regression = leapy.LocalLinearRegression(debug_enable=True, fit_factor=0.03)
    #
    # linear_regression.fit(sample_train, label_train)
    # label_predict = numpy.mat(linear_regression.predict(sample_test))
    #
    # color_map = pyplot.get_cmap('viridis')
    # m1 = pyplot.scatter(sample_train[:, 1].tolist(), label_train.tolist(), color=color_map(0.9), s=10)
    # m2 = pyplot.scatter(sample_test[:, 1].tolist(), label_test.tolist(), color=color_map(0.5), s=10)
    #
    # index = sample_test[:, 1].argsort(0)
    # pyplot.plot(sample_test[index, :][:, 0, :][:, 1], label_predict[index, 0], '-', color="black")
    # pyplot.legend((m1, m2), ("Training handwritten_numeral", "Test handwritten_numeral"), loc='lower right')
    # pyplot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # temperature_data.txt
    # ------------------------------------------------------------------------------------------------------------------
    sample_train, sample_test, label_train, label_test = leapy.load_temperature_data(split_enable=True)

    # linear_regression = leapy.StandardLinearRegression(algorithm="math_solving",
    #                                                    debug_enable=True, threshold_enable=False,
    #                                                    polynomial_degree=10,
    #                                                    max_iteration=10000)
    # linear_regression = leapy.RidgeLinearRegression(algorithm="math_solving",
    #                                                 debug_enable=True, threshold_enable=False,
    #                                                 polynomial_degree=10,
    #                                                 max_iteration=10000)
    linear_regression = leapy.LassoLinearRegression(algorithm="math_solving",
                                                    debug_enable=True,
                                                    threshold_enable=False,
                                                    polynomial_degree=10,
                                                    max_iteration=100)
    # linear_regression = leapy.LocalLinearRegression(fit_factor=0.1)

    linear_regression.fit(sample_train, label_train)
    label_predict = linear_regression.predict(sample_test)

    color_map = pyplot.get_cmap('viridis')
    m1 = pyplot.scatter((300 * sample_train).T.tolist()[0], label_train.T.tolist()[0],
                        color=color_map(0.9), s=10)
    m2 = pyplot.scatter((300 * sample_test).T.tolist()[0], label_test.T.tolist()[0],
                        color=color_map(0.5), s=10)

    index = sample_test[:, 0].argsort(0)
    pyplot.plot((300 * sample_test)[index, 0].T.tolist()[0], label_predict[index, 0].T.tolist()[0],
                color='black', linewidth=2, label="Prediction")

    pyplot.suptitle("Linear Regression")
    pyplot.xlabel('time')
    pyplot.ylabel('temperature')
    pyplot.legend((m1, m2), ("Training handwritten_numeral", "Test handwritten_numeral"), loc='lower right')
    pyplot.show()

    print(linear_regression.predict(0))
