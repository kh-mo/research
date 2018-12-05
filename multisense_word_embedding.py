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
parser.add_argument("--batch_size", help="voca size", type=int, default=4)
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


import random

def make_batch(raw_data, voca, batch_size, window_size):
    result = []
    center_word = random.sample(range(window_size+1, len(raw_data)-window_size), batch_size)

    for i in center_word:
        input_list = []
        vocab = voca.copy()
        input_list.append([vocab.index(raw_data[i])])

        pos = [vocab.index(raw_data[i+window]) for window in range(-window_size, window_size+1) if window !=0]
        for pos_word in pos:
            vocab.remove(vocab[pos_word])
        neg = [vocab.index(i) for i in random.sample(vocab, len(pos))]
        input_list.append(pos)  # pos
        input_list.append(neg) # neg
        result.append(input_list)

    return result


if __name__ == "__main__":
    print("Load tokenized_data.pickle")
    tokenized_data =  pickle.load(open(os.getcwd() + "/data/tokenized_data.pickle", 'rb'))
    print("Load voca_count")
    voca_count = json.load(open(os.getcwd() + "/data/voca_count"))
    voca = list(voca_count.keys()) + ["pad_token", "unk_token"]

    model = word2vec(args.embed_size, args.voca_size)
    batch = make_batch(tokenized_data, voca, args.batch_size, args.window)
    model(batch[0][0])

## loss 추가
## batch 함수 추가