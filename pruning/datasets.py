from torchvision import datasets, transforms

def get_dataset(name, train=False, split="val"):
    dataset = None
    if name == "imagenet":
        dataset = datasets.ImageNet(root="dataset/",
                                    split=split,
                                    download=True,
                                    transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                       std=[0.229, 0.224, 0.225])]))
    elif name == "mnist":
        dataset = datasets.MNIST(root="dataset/",
                                 train=train,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    return dataset