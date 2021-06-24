#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/2 1:13
# @Author   : Mr. Fan

"""

"""

from matplotlib import pyplot
import leapy


if __name__ == "__main__":
    titles = [
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

    model = leapy.LatentSemanticAnalysis()
    model.generate(titles, stop_words=['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to', 's'])
    print(model.word_topic)
    print(model.topic_document)
    model.visualizing()
