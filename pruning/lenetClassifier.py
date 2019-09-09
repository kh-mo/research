import os

import torch
from torch.utils.data import DataLoader

from models import Lenet_300_100, Lenet_5


model_folder = os.path.join(os.getcwd(), "models")
try:
    os.mkdir(model_folder)
except FileExistsError as e:
    pass

##################### model-300-100 #####################

model_300_100 = Lenet_300_100().to(device)
torch.save(model.state_dict(), os.path.join(model_folder, "lenet_300_100"))
######################## model-5 ########################

model_5 = Lenet_5().to(device)
torch.save(model.state_dict(), os.path.join(model_folder, "lenet_5"))
#########################################################


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


##########################################################################################
import os
os.chdir(os.path.join(os.getcwd(), "pruning"))

import torch
import argparse
from models import get_model
from collections import OrderedDict

import torch.nn as nn
from datasets import get_dataset
from torch.utils.data import DataLoader

# def pruning(model, threshold):
#     return pruned_model, pruned_position
#
# def prune_backward_hook_function(module, grad_input, grad_output):
#     # return print("module : {}, grad_input : {}, grad_output : {}".format(module, grad_input, grad_output))
#     modified_grad_out = prune_position_list[-1] * grad_input[0]
#     del prune_position_list[-1]
#     return module, (modified_grad_out,), grad_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    ## model 선언
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, device)
    model = get_model("lenet", device)
    ## retrain위한 함수 설정
    # model.register_backward_hook(prune_backward_hook_function)
    # for position, module in model.classifier._modules.items():
    #     print(module)
    #     module.register_backward_hook(prune_backward_hook_function)

    ## retrain시 필요한 데이터, loss, optim 설정
    train_datasets = DataLoader(get_dataset(name="mnist", train=True), batch_size=256)
    loss_function = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0002)

    prune_model = {}
    prune_position_list = []
    threshold = 0.0

    # pruning 반복 횟수
    for i in range(10):
        # pruning 수행
        for name, tensor in model.state_dict().items():
            prune_position = torch.clamp(tensor, min=threshold)
            prune_position[prune_position > threshold] = 1
            prune_model[name] = tensor * prune_position
            prune_position_list.append(prune_position)

        ## model save
        new_state_dict = OrderedDict(prune_model)
        model.load_state_dict(new_state_dict, strict=False)
        # torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/pruned_{}_times_{}_model".format(i, args.model)))
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/pruned_{}_times_{}_model".format(1, "lenet")))

        loss_list = []
        for input, target in train_datasets:
            # print(input)
            # break
            model_output = model(input.to(device))
            loss = loss_function(model_output, target.to(device))
            optim.zero_grad()
            loss.backward()
            for idx, p in enumerate(model.parameters()):
                p.grad = p.grad * prune_position_list[idx]
            optim.step()
            loss_list.append(loss.item())

        loss = sum(loss_list) / len(loss_list)
        print("{} epoch, loss : {}".format(i, loss))

model.state_dict().items()


p.grad * prune_position_list[-1]

len(prune_position_list)



#### to be
    # total_parameter = get_param(pretrain_model)
    # total_hyperparameter = get_hyper_param(pretrain_model)
    #
    # import os
    # from torchvision.utils import save_image
    # def save_images(image, epoch):
    #     saved_folder = os.path.join(os.getcwd(), "saved_image")
    #     try:
    #         os.mkdir(saved_folder)
    #     except FileExistsError as e:
    #         pass
    #     save_image(image, saved_folder + '/' + str(epoch + 1) + '_epoch_image.png', nrow=16)
    #
    # save_images(test_img, 0)
