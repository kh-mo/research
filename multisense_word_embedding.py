"""
Refer to from word2gm_trainer.py
(https://github.com/benathi/word2gm/blob/master/word2gm_trainer.py)
the model in Multimodal Word Distributions, ACL 2017.
"""
# pytorch로 구현
# 전역변수 설정

import os
import json
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--window", help="window size", type=int, default=5)
args, _ = parser.parse_known_args()

class multisense_model():
    def __init__(self):

    def train(self):
        return
    def test(self):
        return
    def visualization(self):
        # with gephi
        return

def get_batch():
    return

if __name__ == "__main__":
    print("Load tokenized_data.pickle")
    tokenized_data =  pickle.load(open(os.getcwd() + "/data/tokenized_data.pickle", 'rb'))
    print("Load voca_count")
    voca_count = json.load(open(os.getcwd() + "/data/voca_count"))

    model = multisense_model()
    for epoch in range(epoch):
        model.train()
    model.test()
    model.visualization()

