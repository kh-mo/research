import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

class Lenet(nn.Module):
    def __init__(self, num_classes=10):
        super(Lenet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(name, device, pretrain=True):
    model = None
    if name == "alexnet":
        model = models.alexnet(pretrained=pretrain).to(device)
    elif name == "lenet":
        if os.path.isfile(os.path.join(os.getcwd(), "models/lenet-300-100")):
            model = Lenet().to(device)
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models/lenet-300-100")))
        else:
            model = Lenet().to(device)
            from datasets import get_dataset
            train_datasets = DataLoader(get_dataset(name="mnist", train=True), batch_size=256)
            loss_function = nn.CrossEntropyLoss().to(device)
            optim = torch.optim.Adam(model.parameters(), lr=0.0002)
            for i in range(100): ## 100 에폭 훈련
                loss_list = []
                for input, target in train_datasets:
                    loss = loss_function(model(input.to(device)), target.to(device))
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_list.append(loss.item())
                if i % 10 == 0 or i == 99:
                    loss = sum(loss_list) / len(loss_list)
                    print("{} epoch, loss : {}".format(i, loss))
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/lenet-300-100"))
    return model