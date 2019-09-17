'''
step 1. make saved model folder
step 2. get dataset
step 3. training
step 4. saving
'''
import os
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Lenet_300_100, Lenet_5
from datasets import get_dataset

def get_model(args):
    model = None
    if args.model == "lenet_300_100":
        model = Lenet_300_100().to(args.device)
    elif args.model == "lenet_5":
        model = Lenet_5().to(args.device)
    else:
        warnings.warn("{} model does not exist.".format(args.model))
    return model

def training(model, data, args):
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0002)
    for i in range(100):  ## training 100 epoch
        loss_list = []
        for input, target in data:
            loss = loss_function(model(input.to(args.device)), target.to(args.device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_list.append(loss.item())
        if i % 10 == 0 or i == 99:
            loss = sum(loss_list) / len(loss_list)
            print("{} epoch, loss : {}".format(i, loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # step 1
    model_folder = os.path.join(os.getcwd(), "models")
    try:
        os.mkdir(model_folder)
    except FileExistsError as e:
        pass

    # step 2
    model = get_model(args)
    trainset, testset = get_dataset(args)
    train_data = DataLoader(trainset, batch_size=args.batch_size)
    print("{} model, {} dataset load complete!!".format(args.model, args.dataset))

    # step 3
    print("use {} for training".format(args.device))
    training(model, train_data, args)
    print("Finish training")

    # step 4
    torch.save(model.state_dict(), os.path.join(model_folder, "{}".format(args.model)))
    print("Complete model saving")
