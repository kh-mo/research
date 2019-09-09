'''

python pruningRetraining.py --model="alexnet" --dataset="imagenet" --pruning_epoch=10
return : acc, a number of parameters

step 1. get model, dataset
step 2. pruning
step 3. retraining
step 4. evaluate

'''

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_model
from datasets import get_dataset
from utils import evaluate

def pruning(model, args):
    prune_model = {}
    prune_position_list = []
    threshold = args.pruningThreshold

    for name, tensor in model.state_dict().items():
        prune_position = torch.clamp(tensor, min=threshold)
        prune_position[prune_position > threshold] = 1
        prune_model[name] = tensor * prune_position
        prune_position_list.append(prune_position)

    new_state_dict = OrderedDict(prune_model)
    model.load_state_dict(new_state_dict, strict=False)
    return prune_position_list

def retraining(model, train_data, prune_position_list, args):
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learningRate)

    print("start retraining")
    for epoch in range(args.retrainingEpochs):
        loss_list = []
        for input, target in train_data:
            model_output = model(input.to(args.device))
            loss = loss_function(model_output, target.to(args.device))
            optim.zero_grad()
            loss.backward()
            for idx, p in enumerate(model.parameters()):
                p.grad = p.grad * prune_position_list[idx]
            optim.step()
            loss_list.append(loss.item())

        loss = sum(loss_list) / len(loss_list)
        print("{} epoch, loss : {}".format(epoch+1, loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pruningEpochs", type=int, default=10)
    parser.add_argument("--pruningThreshold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learningRate", type=int, default=0.0002)
    parser.add_argument("--retrainingEpochs", type=int, default=10)
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
    for epoch in range(args.pruningEpochs):
        prune_position_list = pruning(model, args)

        # step 3
        retraining(model, train_data, prune_position_list, args)

        # step 4
        evaluate(model, test_data, args)
        # for input, target in test_data:
        #     evaluate(model)
        # save(model)
