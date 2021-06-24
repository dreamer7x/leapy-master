import numpy
import math


# Min-Max 标准化
def min_max(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        x_max = numpy.max(x[:, i])
        x_min = numpy.min(x[:, i])
        distance = x_max - x_min
        for j in range(raw):
            x_convert[j, i] = (x[j, i] - x_min) / distance
    return x_convert


# Z-Score 标准化
def z_score(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        average = sum(x[:, i].T.tolist()[0]) / raw
        standard_deviation = pow(sum([pow(i - average, 2) for i in x[:, i].T.tolist()[0]]) / raw, 1 / 2)
        for j in range(raw):
            x_convert[j, i] = (x[j, i] - average) / standard_deviation
    return x_convert


# Decimal scaling 标准化
def decimal_scaling(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        absolute = numpy.abs(x[:, i])
        absolute_max = numpy.max(absolute[:, i])
        bit = len(str(int(absolute_max)))
        for j in range(raw):
            x_convert[j, i] = x[j, i] / 10 ** bit
    return x_convert


# 均值归一化
def average_normalization(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        average = numpy.mean(x[:, i])
        x_max = numpy.max(x[:, i])
        x_min = numpy.min(x[:, i])
        distance = x_max - x_min
        for j in range(raw):
            x_convert[j, i] = (x[j, i] - average) / distance
    return x_convert


def vector_normalization(x, dimension=None):
    """
    Vector normalization 向量归一化

    Parameters
    ----------
    x : object numpy.matrix shape of N * p
    dimension : list

    Returns
    -------

    """
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    if dimension is None:
        dimension = [i for i in range(col)]

    for i in dimension:
        l2 = numpy.linalg.norm(x[:, i], axis=0, ord=2)
        for j in range(raw):
            x_convert[j, i] = x[j, i] / l2

    return x_convert


# sigmoid 归一化
def sigmoid_normalization(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        for j in range(raw):
            x_convert[j, i] = 1 / (1 + math.exp(-x[j, i]))
    return x_convert


# soft_max 归一化
def soft_max_normalization(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        e_sum = numpy.sum(numpy.exp(x[:, i]))
        for j in range(raw):
            x_convert[j, i] = numpy.exp(x[j, i]) / e_sum
    return x_convert


# lg 归一化
# 要求处理数据大于等于1
def lg_normalization(x):
    x = numpy.mat(x)
    raw, col = x.shape
    x_convert = numpy.mat(numpy.zeros((raw, col)))
    for i in range(col):
        x_max = numpy.max(x[:, i])
        for j in range(raw):
            x_convert[j, i] = math.log10(x[j, i]) / math.log10(x_max)
    return x_convert


if __name__ == "__main__":
    x_test = numpy.mat([[i for i in range(1, 11)]]).T
    print("min_max: \n" + str(min_max(x_test)) + "\n")
    print("z_score: \n" + str(z_score(x_test)) + "\n")
    print("decimal_scaling: \n" + str(decimal_scaling(x_test)) + "\n")
    print("average_normalization: \n" + str(average_normalization(x_test)) + "\n")
    print("vector_normalization: \n" + str(vector_normalization(x_test)) + "\n")
    print("sigmoid_normalization: \n" + str(sigmoid_normalization(x_test)) + "\n")
    print("soft_max_normalization: \n" + str(soft_max_normalization(x_test)) + "\n")
    print("lg_normalization: \n" + str(lg_normalization(x_test)) + "\n")
