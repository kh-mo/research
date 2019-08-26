'''
pretrain model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

'''

import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

if __name__ == "__main__":
    '''
    python baseline.py model="alexnet" test_dataset="imagenet"
    return : acc, a number of parameters
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrain_model = models.alexnet(pretrained=True).to(device)
    pretrain_model.eval()

    test_data = DataLoader(datasets.ImageNet(root="dataset/",
                                             split="val",
                                             download=True,
                                             transform=transforms.Compose([transforms.Resize((256,256)),
                                                                           transforms.CenterCrop(224),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                std=[0.229, 0.224, 0.225])])),
                           batch_size=256)

    pred = []
    label = []

    for idx, (test_img, test_label) in enumerate(test_data):
        pred += torch.argmax(pretrain_model(test_img.to(device)),dim=1).tolist()
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