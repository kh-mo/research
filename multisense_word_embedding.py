"""
Refer to from word2gm_trainer.py
(https://github.com/benathi/word2gm/blob/master/word2gm_trainer.py)
the model in Multimodal Word Distributions, ACL 2017.
"""
# pytorch로 구현
# 전역변수 설정

import os
import json
import torch
import pickle
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--window", help="window size", type=int, default=5)
parser.add_argument("--embed_size", help="embedding size", type=int, default=5)
parser.add_argument("--voca_size", help="voca size", type=int, default=10)
args, _ = parser.parse_known_args()

class word2vec(nn.Module):
    def __init__(self, embed_size, voca_size):
        super(word2vec, self).__init__()
        self.embed_1 = nn.Embedding(voca_size, embed_size)
        self.embed_2 = nn.Linear(embed_size, voca_size)

    def forward(self, x):
        ## x == input_idx
        x = self.embed_1(torch.tensor(x))
        x = self.embed_2(x)
        return x

if __name__ == "__main__":
    print("Load tokenized_data.pickle")
    tokenized_data =  pickle.load(open(os.getcwd() + "/data/tokenized_data.pickle", 'rb'))
    print("Load voca_count")
    voca_count = json.load(open(os.getcwd() + "/data/voca_count"))

    model = word2vec(args.embed_size, args.voca_size)
    model(5)

## loss 추가
## batch 함수 추가