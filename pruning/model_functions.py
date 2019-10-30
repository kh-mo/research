from torchvision import models

def get_model(args, pretrain=True):
    model = None
    if args.model == "resnet18":
        model = models.resnet18(pretrained=pretrain).to(args.device)
    elif args.model == "resnet34":
        model = models.resnet34(pretrained=pretrain).to(args.device)
    elif args.model == "alexnet":
        model = models.alexnet(pretrained=pretrain).to(args.device)
    else:
        print("{} model does not exist.".format(args.model))

    if args.do_training == "True":
        model.train()
    else:
        model.eval()
    print("model download from pytorch hub(https://pytorch.org/docs/stable/torchvision/models.html)")
    return model

