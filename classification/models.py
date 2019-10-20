'''
All models can be classified into two categoroies.
1. download from pytorch hub
2. made by author
'''

import os
import math
import warnings

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as f
from torch.nn import Linear

class Lenet(nn.Module):
    def __init__(self, num_classes=10):
        super(Lenet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(args, pretrain=True):
    model = None
    if args.model == "alexnet":
        model = models.alexnet(pretrained=pretrain).to(args.device)
        print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    elif args.model == "vggnet":
        model = models.vgg16(pretrained=pretrain).to(args.device)
        print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    elif args.model == "resnet18":
        model = models.resnet18(pretrained=pretrain).to(args.device)
        print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    elif args.model == "resnet34":
        model = models.resnet34(pretrained=pretrain).to(args.device)
        print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    elif args.model == "lenet":
        model = Lenet().to(args.device)
        # pretrain lenet 사용 시 다시 고려(19.10.09)
        # if check_model(args.model):
        #     model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models/{}".format(args.model))))
        # else:
        #     warnings.warn("pretrained {} model does not exist.".format(args.model))
    elif args.model == "scnn":
        model = SCNN().to(args.device)
    else:
        warnings.warn("{} model does not exist.".format(args.model))

    if args.do_training:
        model.train()
    else:
        model.eval()

    return model

# pretrain lenet 사용 시 다시 고려(19.10.09)
# def check_model(model_name):
#     return os.path.isfile(os.path.join(os.getcwd(), "models/{}".format(model_name)))

class Gaussian(object):
    def __init__(self, mu, rho, args):
        super(Gaussian, self).__init__()
        self.args = args
        self.mu = mu.to(args.device)
        self.rho = rho.to(args.device)
        self.normal = torch.distributions.normal.Normal(loc=0, scale=1)
        self.sigma = 0

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.args.device)
        self.sigma = torch.log(1 + torch.exp(self.rho))
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - (((input - self.mu)/self.sigma)**2)/2).sum()

class ScaleMixturePrior(object):
    def __init__(self, pi, sigma1, sigma2, args):
        super(ScaleMixturePrior, self).__init__()
        self.pi = pi
        self.sigma1 = torch.FloatTensor([math.exp(sigma1)]).to(args.device)
        self.sigma2 = torch.FloatTensor([math.exp(sigma2)]).to(args.device)
        self.gaussian1 = torch.distributions.normal.Normal(0, self.sigma1)
        self.gaussian2 = torch.distributions.normal.Normal(0, self.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(BayesianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-4., 4.))
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-4., 4.))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, args)
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-4., 4.))
        self.bias_rho = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-4., 4.))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, args)

        self.log_variational_posterior = 0
        self.log_prior = 0

        self.pi = 0.5
        self.sigma1 = torch.exp(torch.tensor(-1.)).item()
        self.sigma2 = torch.exp(torch.tensor(-6.)).item()
        self.weight_prior = ScaleMixturePrior(self.pi, self.sigma1, self.sigma2, args)
        self.bias_prior = ScaleMixturePrior(self.pi, self.sigma1, self.sigma2, args)

    def forward(self, input):
        if self.training:
            weight = self.weight.sample()
            bias = self.bias.sample()
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        else:
            weight = self.weight.mu
            bias = self.bias.mu
            self.log_variational_posterior = 0
            self.log_prior = 0
        return f.linear(input, weight, bias)

def get_bayesian(model, args):
    for name, module in model._modules.items():
        if isinstance(module, Linear):
            print("name :", name)
            print("module :", module)
            linear_layer = model._modules[name]
            model._modules[name] = BayesianLinear(linear_layer.in_features, linear_layer.out_features, args)
            continue
        for sub_name, sub_module in module._modules.items():
            if isinstance(sub_module, Linear):
                print("sub_name :", sub_name)
                print("sub_module :", sub_module)
                linear_layer = model._modules[name]._modules[sub_name]
                model._modules[name]._modules[sub_name] = BayesianLinear(linear_layer.in_features, linear_layer.out_features, args)
    return model

