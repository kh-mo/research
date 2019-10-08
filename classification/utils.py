import torch
import numpy as np
import torch.nn as nn

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
