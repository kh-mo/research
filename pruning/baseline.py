'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

python baseline.py --model="alexnet" --dataset="imagenet"
return : acc, a number of parameters

Lenet is not in pytorch hub so we make it.

step 1. get model, dataset
step 2. evaluate
'''

import argparse

import torch
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset
from utils import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # step 1
    model = get_model(args)
    print("{} model load complete!!".format(args.model))

    trainset, testset = get_dataset(args)
    train_data = DataLoader(trainset, batch_size=args.batch_size)
    test_data = DataLoader(testset, batch_size=args.batch_size)
    print("{} dataset load complete!!".format(args.dataset))

    # step 2
    print("use {} for evaluating".format(args.device))
    evaluate(model, test_data, args)
