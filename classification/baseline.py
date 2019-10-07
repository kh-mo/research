'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

step 1. get model, dataset
step 2. modify model
step 3. training(option)
step 4. evaluate
step 5. save model
'''

import os
import argparse

import torch
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset
from utils import evaluate, modify_model, training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--do_training", type=bool, default=True)
    parser.add_argument("--learningRate", type=int, default=0.0001)
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
    model = modify_model(model, args)
    print("modify model done.")

    # step 3
    if model.training:
        print("start training")
        training(model, train_data, args)
        print("training done.")

    # step 4
    print("use {} for evaluating".format(args.device))
    acc, param_count = evaluate(model, test_data, args)

    # step 5
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/{}_{}_acc_{}".format(args.model, args.dataset, acc)))
    print("Complete model saving")