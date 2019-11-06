import os
import time
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
        print("training by {}, {} epoch, loss : {}".format(args.dataset, epoch + 1, loss))

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
    total_parameter = 0
    nonzero_parameter = 0
    for i in model.parameters():
        total_parameter += torch.flatten(i, 0).shape[0]
        nonzero_parameter += len(np.nonzero(i.detach().cpu().numpy())[0])

    print("Accuracy : {}, Total_Parameters : {}, Nonzero-Parameter : {}".format(total_acc, total_parameter, nonzero_parameter))
    return total_acc

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

def check_inference_time(model, dataset, args):
    inference_time = []
    dataiter = iter(dataset)
    for i in range(args.inference_sampling):
        image, target = dataiter.next()
        inference_start_time = time.time()
        model_output = model(image.to(args.device))
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        inference_time.append(round(total_inference_time, 3)*1000)
    return inference_time
