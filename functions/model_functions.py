import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

class Lenet_300_100(nn.Module):
    def __init__(self):
        super(Lenet_300_100, self).__init__()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(28 * 28, 300)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(300, 100)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(100, 10)),
        ]))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Lenet_5(nn.Module):
    def __init__(self):
        super(Lenet_5, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, kernel_size=3)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(6, 6, kernel_size=3)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout()),
            ('fc1', nn.Linear(216, 128)),
            ('relu1', nn.ReLU()),
            ('drop2', nn.Dropout()),
            ('fc2', nn.Linear(128, 10))
        ]))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(args, pretrain=True):
    model = None
    if args.model == "resnet18":
        model = models.resnet18(pretrained=pretrain).to(args.device)
    elif args.model == "resnet34":
        model = models.resnet34(pretrained=pretrain).to(args.device)
    elif args.model == "alexnet":
        model = models.alexnet(pretrained=pretrain).to(args.device)
    elif args.model == "lenet_300_100":
        model = Lenet_300_100().to(args.device)
    elif args.model == "lenet_5":
        model = Lenet_5().to(args.device)
    else:
        print("{} model does not exist.".format(args.model))

    if args.load_folder_model != "None":
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models/{}".format(args.load_folder_model))))

    if args.do_training == "True":
        model.train()
    else:
        model.eval()
    print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    return model

