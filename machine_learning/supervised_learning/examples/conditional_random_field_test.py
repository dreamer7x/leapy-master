#!/usr/bin/env python

import argparse
import leapy

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("data_file", help="handwritten_numeral file for testing input")
    # parser.add_argument("model_file", help="the model file name.")
    #
    # args = parser.parse_args()
    #
    # model = leapy.LinearChainConditionalRandomField()
    # model.load_model(args.model_file)
    # model.score(args.data_file)

    train_data, test_data = leapy.load_chunking_data()

    model = leapy.LinearChainConditionalRandomField(debug_enable=True)
    model.fit(data=train_data, model_path="models/linear_chain_conditional_random_model.json")

    model = leapy.LinearChainConditionalRandomField(debug_enable=True)
    model.score(data=test_data, model_path="models/linear_chain_conditional_random_model.json")
