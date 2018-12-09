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
import random
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

parser = argparse.ArgumentParser()
parser.add_argument("--window", help="window size", type=int, default=5)
parser.add_argument("--embed_size", help="embedding size", type=int, default=300)
parser.add_argument("--voca_size", help="voca size", type=int, default=20000)
parser.add_argument("--batch_size", help="voca size", type=int, default=32)
parser.add_argument("--epsilon", help="voca size", type=int, default=0.00001)
args, _ = parser.parse_known_args()

class word2vec(nn.Module):
    def __init__(self, embed_size, voca_size):
        super(word2vec, self).__init__()
        self.embed_1 = nn.Embedding(voca_size, embed_size)
        self.embed_2 = nn.Linear(embed_size, voca_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ## x == input_idx
        x = self.embed_1(torch.tensor(x))
        x = self.embed_2(x)
        # x = self.softmax(x)
        return x

def make_batch(raw_data, voca, special_token, batch_size, window_size):
    center_word_list = random.sample(range(window_size+1, len(raw_data)-window_size), batch_size)
    center_word_idx_list = []
    pos_list = []
    neg_list = []

    for i in center_word_list:
        center_word = raw_data[i]
        if center_word not in voca:
            continue
        center_word_idx = voca.index(center_word)
        count = 0
        vocab = list(range(len(voca)))
        for window in range(-window_size, window_size+1):
            if window == 0:
                continue
            center_word_idx_list += [center_word_idx]
            window_word = raw_data[i+window]
            if window_word in voca:
                pos = voca.index(window_word)
                pos_list += [pos]
                count += 1
                try:
                    vocab.remove(pos)
                except ValueError as e:
                    pass
            else:
                pos = special_token["unk_token"]
                pos_list += [pos]
                count += 1
        neg_list += random.sample(vocab, count)

    result = [center_word_idx_list, pos_list, neg_list]
    return result

def most_similarity_vector(word, number):
    embed_mat = model.embed_1(torch.tensor(range(args.voca_size)))
    input_word_idx = voca.index(word)
    input_emb = embed_mat[input_word_idx]

    numerator = torch.sum(torch.mul(torch.unsqueeze(input_emb, dim=0), embed_mat), dim=1)
    denominator = torch.sqrt(torch.unsqueeze(torch.sum(torch.mul(input_emb,input_emb)),dim=0)) * torch.sqrt(torch.sum(torch.mul(embed_mat,embed_mat), dim=1))

    cos_emb = numerator / denominator

    sorted_list = sorted(range(len(cos_emb)), key=lambda i: cos_emb[i], reverse=True)[:number+1]
    return [{voca[idx]:cos_emb[idx].item()} for idx in sorted_list[1:]]

if __name__ == "__main__":
    print("Load tokenized_data.pickle")
    tokenized_data =  pickle.load(open(os.getcwd() + "/data/tokenized_data.pickle", 'rb'))
    print("Load voca_count")
    voca_count = json.load(open(os.getcwd() + "/data/voca_count"))

    sorted_voca_count = sorted(voca_count.items(), key=lambda x: x[1])
    sorted_voca_count = sorted_voca_count[len(sorted_voca_count)-args.voca_size+2:]

    voca = [word for word, count in sorted_voca_count]
    special_token = {"pad_token":args.voca_size-2, "unk_token":args.voca_size-1}

    model = word2vec(args.embed_size, args.voca_size)#.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    before_embedding_vector = most_similarity_vector("apple", 10)
    tmp1 = torch.sum(model.embed_1(torch.tensor([1])))

    for i in range(200):
        batch = make_batch(tokenized_data, voca, special_token, args.batch_size, args.window)
        optimizer.zero_grad()

        model_output = model(batch)
        pred = model_output[0]
        pos_label = model_output[1]
        neg_label = model_output[2]
        '''
        pred = model(batch[0])
        pos_label = model(batch[1])
        neg_label = model(batch[2])
        '''
        pos_loss = -torch.log(torch.sigmoid(torch.sum(torch.mul(pred, pos_label), dim=1))+args.epsilon)
        neg_loss = -torch.log(torch.sigmoid(torch.sum(torch.mul(pred, -neg_label), dim=1))+args.epsilon)

        total_loss = torch.sum(pos_loss + neg_loss)
        if i % 10 == 0 :
            print(total_loss)

        total_loss.backward()
        optimizer.step()

    after_embedding_vector = most_similarity_vector("apple", 10)

'''
for param in model.parameters():
    print(param.grad)
    print(param.grad.data.sum())

total_loss.grad_fn.next_functions[0][0]
'''
neg_label.grad_fn.next_functions[0][0].next_functions[0][0]
pred.grad_fn.next_functions[0][0].next_functions[0][0]
pos_label.grad_fn.next_functions[0][0].next_functions[0][0]
model_output.grad_fn.next_functions