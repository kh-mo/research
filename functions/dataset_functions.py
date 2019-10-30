import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset(args):
    dataset_folder = os.path.join(os.getcwd(), "datasets")
    try:
        os.mkdir(dataset_folder)
    except FileExistsError as e:
        print("{} folder already exist".format(args.dataset))
        pass

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
    else:
        print("{} dataset does not exist.".format(args.dataset))

    train_data = DataLoader(train, batch_size=args.batch_size)
    test_data = DataLoader(test, batch_size=args.batch_size)
    dataset = (train_data, test_data)
    return dataset