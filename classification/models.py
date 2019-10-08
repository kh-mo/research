'''
All models can be classified into two categoroies.
1. download from pytorch hub
2. made by author
'''

import os
import warnings

import torch
import torch.nn as nn
from torchvision import models

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
        if check_model(args.model):
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models/{}".format(args.model))))
        else:
            warnings.warn("pretrained {} model does not exist.".format(args.model))
    else:
        warnings.warn("{} model does not exist.".format(args.model))

    if args.do_training:
        model.train()
    else:
        model.eval()

    return model

def check_model(model_name):
    return os.path.isfile(os.path.join(os.getcwd(), "models/{}".format(model_name)))