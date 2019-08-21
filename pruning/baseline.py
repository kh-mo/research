'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

'''

from torchvision import models, datasets

if __name__ == "__main__":
    pretrain_model = models.alexnet(pretrained=True)

    test_data = datasets.ImageNet(root="dataset/", split="val", download=True)
    test_label = ????

    pred = pretrain_model(test_data)


    total_acc = match(pred, test_label)
    total_parameter = get_param(pretrain_model)
    total_hyperparameter = get_hyper_param(pretrain_model)