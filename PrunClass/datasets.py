import os
from torchvision import datasets, transforms

def get_dataset(args):
    dataset_folder = os.path.join(os.getcwd(), "datasets")
    try:
        os.mkdir(dataset_folder)
    except FileExistsError as e:
        pass
    dataset = None

    if args.dataset == "cifar10":
        train = datasets.CIFAR10(root=dataset_folder,
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.Resize((224,224)),
                                                               transforms.ToTensor()]))
        test = datasets.CIFAR10(root=dataset_folder,
                                train=False,
                                download=True,
                                transform=transforms.Compose([transforms.Resize((224,224)),
                                                              transforms.ToTensor()]))
        dataset = (train, test)

    else:
        print("{} dataset does not exist.".format(args.dataset))

    return dataset