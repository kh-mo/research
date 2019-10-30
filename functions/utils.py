import os
import torch
import numpy as np
import torch.nn as nn

def training(model, train_dataset, args):
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.learning_rate / args.epochs)

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

def evaluating(model, test_data, args):
    # get model accuracy
    pred = []
    label = []
    model.eval()
    for test_img, test_label in test_data:
        pred += torch.argmax(model(test_img.to(args.device)), dim=1).tolist()
        label += test_label.tolist()
    total_acc = sum([1 if pred[i] == label[i] else 0 for i in range(len(pred))]) / len(pred)

    # get number of parameters
    count = 0
    for i in model.parameters():
        count += len(np.nonzero(i.detach().cpu().numpy())[0])

    print("Accuracy : {}, Parameters : {}".format(total_acc, count))
    return total_acc, count

def saving(model, args):
    try:
        os.mkdir(os.path.join(os.getcwd(), "models"))
    except FileExistsError as e:
        print("{} folder already exist".format(args.dataset))
        pass

    if args.load_folder_model == "None":
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/{}_{}_acc_{}_epoch_{}".format(
            args.model, args.dataset, args.accuracy, args.epochs)))
    else:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/{}_{}_{}_acc_{}_epoch_{}".format(
            args.model, args.pruning_method, args.dataset, args.accuracy, args.epochs)))