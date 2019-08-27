'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html
'''

import argparse
import torch
from torch.utils.data import DataLoader

def get_model(name, device, pretrain=True):
    from torchvision import models
    model = None
    if name == "alexnet":
        model = models.alexnet(pretrained=pretrain).to(device)
    return model

def get_dataset(name):
    from torchvision import datasets, transforms
    dataset = None
    if name == "imagenet":
        dataset = datasets.ImageNet(root="dataset/",
                                    split="val",
                                    download=True,
                                    transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                       std=[0.229, 0.224, 0.225])]))
    return dataset

if __name__ == "__main__":
    '''
    python baseline.py model="alexnet" test_dataset="imagenet"
    return : acc, a number of parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain_model = get_model(args.model, args.device, pretrain=True)
    pretrain_model.eval()
    print("{} model load complete!!".format(args.model))

    test_data = DataLoader(get_dataset(args.test_dataset), batch_size=args.batch_size)
    print("{} dataset load complete!!".format(args.test_dataset))

    pred = []
    label = []
    for idx, (test_img, test_label) in enumerate(test_data):
        pred += torch.argmax(pretrain_model(test_img.to(args.device)), dim=1).tolist()
        label += test_label.tolist()
        print(idx)

    total_acc = sum([1 if pred[i] == label[i] else 0 for i in range(len(pred))]) / len(pred)
    print(total_acc)
    total_parameter = get_param(pretrain_model)
    total_hyperparameter = get_hyper_param(pretrain_model)

    import os
    from torchvision.utils import save_image
    def save_images(image, epoch):
        saved_folder = os.path.join(os.getcwd(), "saved_image")
        try:
            os.mkdir(saved_folder)
        except FileExistsError as e:
            pass
        save_image(image, saved_folder + '/' + str(epoch + 1) + '_epoch_image.png', nrow=16)

    save_images(test_img, 0)