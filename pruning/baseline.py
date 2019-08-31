'''

pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

python baseline.py --model="alexnet" --dataset="imagenet"
return : acc, a number of parameters

Lenet is not in pytorch hub so we have to train first and then check that performance.

'''

import argparse
import torch
from torch.utils.data import DataLoader
from models import get_model
from datasets import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain_model = get_model(args.model, args.device, pretrain=True)
    pretrain_model.eval()
    print("{} model load complete!!".format(args.model))

    test_data = DataLoader(get_dataset(args.dataset), batch_size=args.batch_size)
    print("{} dataset load complete!!".format(args.dataset))

    pred = []
    label = []
    for idx, (test_img, test_label) in enumerate(test_data):
        pred += torch.argmax(pretrain_model(test_img.to(args.device)), dim=1).tolist()
        label += test_label.tolist()
        print(idx)

    total_acc = sum([1 if pred[i] == label[i] else 0 for i in range(len(pred))]) / len(pred)
    print(total_acc)


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