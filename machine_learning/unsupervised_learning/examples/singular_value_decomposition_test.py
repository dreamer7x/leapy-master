#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/31 21:39
# @Author   : Mr. Fan

"""

"""

import leapy

if __name__ == "__main__":
    train_data = leapy.load_matrix()

    model = leapy.SingularValueDecomposition()
    model.generate(train_data / 255.0)
    test_data = model.rebuild() * 255.0

    print(train_data)
    print(test_data)
