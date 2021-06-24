#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/5 1:37
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot

import leapy


if __name__ == "__main__":
    documents = [
        "The Neatest Little Guide to Stock Market Investing",
        "Investing For Dummies, 4th Edition",
        "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
        "The Little Book of Value Investing",
        "Value Investing: From Graham to Buffett and Beyond",
        "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
        "Investing in Real Estate, 5th Edition",
        "Stock Investing For Dummies",
        "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors "
        "Miss"
    ]
    model = leapy.ProbabilisticLatentSemanticAnalysis(debug_enable=True)
    model.generate(documents, iteration=10, stop_words=['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to', 's'])
    print(model.word_topic)
    print(model.topic_document)

    pyplot.scatter(model.topic_document[0, :].tolist(), model.topic_document[1, :].tolist())
    pyplot.show()
