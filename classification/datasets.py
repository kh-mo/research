import os
import warnings

from torchvision import datasets, transforms

def get_dataset(args):
    dataset_folder = os.path.join(os.getcwd(), "datasets")
    try:
        os.mkdir(dataset_folder)
    except FileExistsError as e:
        pass
    dataset = None

    if args.dataset == "imagenet":
        train = datasets.ImageNet(root=dataset_folder,
                                  split="train",
                                  download=True,
                                  transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                transforms.CenterCrop(224),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        val = datasets.ImageNet(root=dataset_folder,
                                split="val",
                                download=True,
                                transform=transforms.Compose([transforms.Resize((256, 256)),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
        dataset = (train, val)

    elif args.dataset == "mnist":
        train = datasets.MNIST(root=dataset_folder,
                               train=True,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))
        test = datasets.MNIST(root=dataset_folder,
                              train=False,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))
        dataset = (train, test)

    elif args.dataset == "cifar10":
        train = datasets.CIFAR10(root=dataset_folder,
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
        test = datasets.CIFAR10(root=dataset_folder,
                                train=False,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
        dataset = (train, test)

    else:
        warnings.warn("{} dataset does not exist.".format(args.dataset))

    return dataset