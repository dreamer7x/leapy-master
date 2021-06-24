#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/5/24 10:07
# @Author   : Mr. Fan

"""
learning_processing.machine_learning
====================================

leapy的机器学习模块
"""

from .supervised_learning.linear_regression.standard_linear_regression import StandardLinearRegression
from .supervised_learning.linear_regression.ridge_linear_regression import RidgeLinearRegression
from .supervised_learning.linear_regression.lasso_linear_regression import LassoLinearRegression
from .supervised_learning.linear_regression.local_linear_regression import LocalLinearRegression

from .supervised_learning.perceptron.perceptron import Perceptron

from .supervised_learning.k_nearest_neighbor.k_nearest_neighbor import KNearestNeighbor

from .supervised_learning.naive_bayesian_classifier.naive_bayesian_classifier import NaiveBayesianClassifier

from .supervised_learning.decision_tree.iterative_dichotomiser_3_tree import IterativeDichotomiser3Tree
from .supervised_learning.decision_tree.classification_tree import ClassificationTree
from .supervised_learning.decision_tree.regression_tree import RegressionTree

__all__ = ['StandardLinearRegression',
           'RidgeLinearRegression',
           'LassoLinearRegression',
           'LocalLinearRegression',
           'Perceptron',
           'KNearestNeighbor',
           # 朴素贝叶斯
           'NaiveBayesianClassifier',
           # 决策树
           'IterativeDichotomiser3Tree',
           'ClassificationTree',
           'RegressionTree']
