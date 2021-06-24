import numpy
import leapy


def shuffle(sample, label, seed=None):
    """
    打乱数据

    Parameters
    ----------
    sample : object numpy.ndarray | shape of N * M
        样本矩阵，N 行样本，M 列维度
    label : object numpy.ndarray | shape of N * 1
        标签矩阵，N 行标签
    seed : int
        随机种子

    Returns
    -------
    sample : object numpy.ndarray
    label : object numpy.ndarray
    """
    if seed:
        numpy.random.seed(seed)
    raw, col = sample.shape
    index = numpy.arange(raw)
    numpy.random.shuffle(index)
    return sample[index, :], label[index, :]


# 将数据拆分成k份训练/测试数据
def cross_validation(sample, label, k=10, is_shuffle=True):
    """
    生成交叉验证数据集

    Parameters
    ----------
    sample : object numpy.ndarray | shape of N * M
        样本矩阵，N 行样本，M 列维度
    label : object numpy.ndarray | shape of N * 1
        标签矩阵，N 行标签
    k : int
        拆分数
    is_shuffle :

    Returns
    -------

    """
    sample = numpy.mat(sample)
    raw, col = sample.shape
    label = numpy.mat(label)
    if label.shape[0] == 1:
        label = label.T
    if label.shape[0] != raw:
        print('------------------------------------------------')
        print('* Error from Classification.fit()             \n'
              '  The number of samples and tags are different. ')
        print('------------------------------------------------')
        return None

    # 是否打乱数据
    if is_shuffle:
        sample, label = shuffle(sample, label)
    raw, col = sample.shape
    left_sample = None
    left_label = None
    left_index = raw % k

    # 是否有剩余样本
    if left_index != 0:
        # 取超出部分样本
        left_sample = sample[-left_index:, :]
        left_label = label[-left_index:, :]
        # 取完整样本
        sample = sample[:-left_index, :]
        label = label[:-left_index, :]

    # 将样本，标签分成k份
    sample_split = numpy.split(sample, k)
    label_split = numpy.split(label, k)
    data_split = []
    for i in range(k):
        sample_test, label_test = sample_split[i], label_split[i]
        sample_train = numpy.concatenate(sample_split[:i] + sample_split[i + 1:], axis=0)
        label_train = numpy.concatenate(label_split[:i] + label_split[i + 1:], axis=0)
        data_split.append([sample_train, sample_test, label_train, label_test])

    if left_index != 0:
        numpy.append(data_split[-1][0], left_sample, axis=0)
        numpy.append(data_split[-1][2], left_label, axis=0)

    return data_split


def train_test_split(sample, label, test_size=0.5, is_shuffle=True, seed=None):
    """
    将数据进行简单切分为训练集和测试集

    Parameters
    ----------
    sample : numpy.ndarray 样本
    label : numpy.ndarray 标签
    test_size : float 抽样比例
    is_shuffle : bool 是否打乱
    seed : int 随机种子

    Returns
    -------

    """
    sample = numpy.mat(sample)
    raw, col = sample.shape
    label = numpy.mat(label)
    if label.shape[0] == 1:
        label = label.T
    if label.shape[0] != raw:
        print('------------------------------------------------')
        print('* Error from train_test_split()               \n'
              '  The number of samples and tags are different. ')
        print('------------------------------------------------')
        return None

    if is_shuffle:
        sample, label = shuffle(sample, label, seed)
    # 按比例切割数据
    split_index = label.shape[0] - int(label.shape[0] // (1 / test_size))
    sample_train, sample_test = sample[:split_index, :], sample[split_index:, :]
    label_train, label_test = label[:split_index, :], label[split_index:, :]

    return sample_train, sample_test, label_train, label_test


if __name__ == "__main__":
    s, l = leapy.load_heart_disease_data()
    split = cross_validation(s, l)
    print(len(split))
    print(split[0])
