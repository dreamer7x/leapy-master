import numpy

import leapy

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # 李航教材
    # ------------------------------------------------------------------------------------------------------------------
    sample, label = leapy.load_loan_data()

    model = leapy.MaximumEntropy(debug_enable=True, max_iteration=10)
    model.fit(sample, label)

    print("答案: " + str(model.predict(numpy.mat([[1, 1, 0, 2]]))[0, 0]))

    # ------------------------------------------------------------------------------------------------------------------
    # heart_disease_data.csv
    # ------------------------------------------------------------------------------------------------------------------
    # sample_train, sample_test, label_train, label_test = leapy.load_heart_disease_data(split_enable=True)
    #
    # model = leapy.MaximumEntropy(max_iteration=100)
    # model.fit(sample_train, label_train)
    #
    # print(model.score(sample_test, label_test))

    # ------------------------------------------------------------------------------------------------------------------
    # credit_card_data.csv
    # ------------------------------------------------------------------------------------------------------------------
    # sample_train, sample_test, label_train, label_test = \
    #     leapy.load_credit_card_data(split_enable=True)
    #
    # model = leapy.MaximumEntropy()
    # model.fit(sample_train, label_train, iteration=100)
    #
    # print(model.score(sample_test, label_test))
