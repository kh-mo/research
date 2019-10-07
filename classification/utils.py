import torch
import numpy as np
import torch.nn as nn

def modify_model(model, args):
    dataset_label_nums = {"mnist": 10, "cifar10": 10}
    dataset_label_num = dataset_label_nums[args.dataset]
    image_dims = {"mnist": 28, "cifar10": 32}
    image_dim = image_dims[args.dataset]
    image_channels = {"mnist": 1, "cifar10": 3}
    image_channel = image_channels[args.dataset]

    models_out_feature_num = list(model.classifier._modules.items())[-1][1].out_features
    print("{} model's output class is {}, {} dataset class number are {}".format(args.model, models_out_feature_num,
                                                                                 args.dataset, dataset_label_num))

    if models_out_feature_num == dataset_label_num:
        print("we do not need training")
        model.eval()
    else:
        print("we need to training")
        model = modify(model, image_dim, image_channel, dataset_label_num, models_out_feature_num, args)
        model.train()
    return model

def modify(model, image_dim, image_channel, dataset_label_num, models_out_feature_num, args):
    new_model = nn.Sequential(up_scaling(image_dim, image_channel).to(args.device),
                              model.to(args.device),
                              new_layer(models_out_feature_num, dataset_label_num).to(args.device))
    return new_model

class new_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(new_layer, self).__init__()
        self.insert_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        x = self.insert_classifier(x)
        return x

class up_scaling(nn.Module):
    def __init__(self, input_dim, input_channel):
        super(up_scaling, self).__init__()
        self.insert_features = nn.Sequential(
            torch.nn.ConvTranspose2d(input_channel, 3, kernel_size=(224-input_dim+1,224-input_dim+1)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.insert_features(x)
        return x

def training(model, train_dataset, args):
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learningRate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.learningRate / args.epochs)

    for epoch in range(args.epochs):
        loss_list = []
        for input, target in train_dataset:
            model_output = model(input.to(args.device))
            loss = loss_function(model_output, target.to(args.device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_list.append(loss.item())

        loss = sum(loss_list) / len(loss_list)
        scheduler.step(epoch+1)
        print("retraining by {}, {} epoch, loss : {}".format(args.dataset, epoch + 1, loss))

def evaluate(model, test_data, args):
    # get model accuracy
    pred = []
    label = []
    for test_img, test_label in test_data:
        pred += torch.argmax(model(test_img.to(args.device)),dim=1).tolist()
        label += test_label.tolist()
    total_acc = sum([1 if pred[i] == label[i] else 0 for i in range(len(pred))]) / len(pred)

    # get number of parameters
    count = 0
    for i in model.parameters():
        count += len(np.nonzero(i.detach().cpu().numpy())[0])

    print("Accuracy : {}, Parameters : {}".format(total_acc, count))
    return total_acc, count
