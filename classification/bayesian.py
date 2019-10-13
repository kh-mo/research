'''
pretrain model을 가져온 다음, bayesian 방식으로 변화시키는 로직을 구현해야 함.

pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

step 1. get model, dataset
step 2. training(option)
step 3. evaluate
step 4. save model
'''
#
# import os
# import argparse
#
# import torch
# from torch.utils.data import DataLoader
#
# from models import get_model, transfer_bayesian
# from datasets import get_dataset
# from utils import evaluate, training
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str)
#     parser.add_argument("--dataset", type=str)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--batch_size", type=int, default=256)
#     parser.add_argument("--do_training", type=bool, default=True)
#     parser.add_argument("--learningRate", type=int, default=0.001)
#     args = parser.parse_args()
#     args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # step 1
#     model = get_model(args)
#     print("{} model load complete!!".format(args.model))
#
#     trainset, testset = get_dataset(args)
#     train_data = DataLoader(trainset, batch_size=args.batch_size)
#     test_data = DataLoader(testset, batch_size=args.batch_size)
#     print("{} dataset load complete!!".format(args.dataset))
#
#     model = transfer_bayesian(model)
#
#     # step 2
#     if model.training:
#         print("start training")
#         training(model, train_data, args)
#         print("training done.")
#
#     # step 3
#     print("use {} for evaluating".format(args.device))
#     acc, param_count = evaluate(model, test_data, args)
#
#     # step 4
#     torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/{}_{}_acc_{}_epoch_{}".format(args.model, args.dataset, acc, args.epochs)))
#     print("Complete model saving")


########################################################

import torch
import torch.nn as nn
import torch.nn.functional as f

class Gaussian(object):
    def __init__(self, mu, rho):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.normal.Normal(loc=0, scale=1)

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        sigma = torch.log(1 + torch.exp(self.rho))
        return self.mu + sigma * epsilon

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BayesianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.2, 0.2))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-0.2, 0.2))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, input):
        print(input) #3
        print(input.shape) #4
        weight = self.weight.sample()
        bias = self.bias.sample()
        print(weight.shape)
        print(bias.shape)
        return f.linear(input, weight, bias)

class bayesian(nn.Module):
    def __init__(self, num_classes=10):
        super(bayesian, self).__init__()
        self.classifier = nn.Sequential(
            BayesianLinear(784, 10),
        )
    def forward(self, x):
        x = torch.flatten(x, 1) #1
        x = self.classifier(x) #2
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input = torch.randn(2,1,28,28)
target = torch.randn(2,10)
model = bayesian().to(device)
model(input.to(device))

loss 구하기