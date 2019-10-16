'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

step 1. get model, dataset, get_bayesian(option)
step 2. training(option)
step 3. evaluate
step 4. save model
'''

import os
import argparse

import torch
from torch.utils.data import DataLoader

from models import get_model, get_bayesian
from datasets import get_dataset
from utils import evaluate, training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--do_training", type=bool, default=True)
    parser.add_argument("--learningRate", type=int, default=0.001)
    parser.add_argument("--do_bayesian", type=bool, default=False)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # step 1
    model = get_model(args)
    print("{} model load complete!!".format(args.model))
    if args.do_bayesian:
        model = get_bayesian(model, args)
        print("change linear layer to bayesian linear layer")

    trainset, testset = get_dataset(args)
    train_data = DataLoader(trainset, batch_size=args.batch_size)
    test_data = DataLoader(testset, batch_size=args.batch_size)
    print("{} dataset load complete!!".format(args.dataset))

    # step 2
    if model.training:
        print("start training")
        training(model, train_data, args)
        print("training done.")

    # step 3
    print("use {} for evaluating".format(args.device))
    acc, param_count = evaluate(model, test_data, args)

    # step 4
    if args.do_bayesian:
        torch.save(model.state_dict(), os.path.join(os.getcwd(),"models/bayesian_{}_{}_acc_{}_epoch_{}".format(args.model, args.dataset, acc,args.epochs)))
    else:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/{}_{}_acc_{}_epoch_{}".format(args.model, args.dataset, acc, args.epochs)))
    print("Complete model saving")