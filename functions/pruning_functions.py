from collections import OrderedDict

import torch
import torch.nn as nn

def pruning(model, train_data, args):
    if args.pruning_method == "songhan":
        print("run songhan algorithm")
        songhan_algorithm(model, train_data, args)
    elif args.pruning_method == "fpgm":
        print("run fpgm algorithm")
    else:
        print("{} algorithm does not exist.".format(args.pruning_method))

def songhan_algorithm(model, train_data, args):
    cut_point = get_cut_point(model, args.cut_rate)
    for epoch in range(args.pruning_epochs):
        print("start functions {} epoch\n".format(epoch + 1))
        prune_position_list = run_pruning(model, cut_point)
        retraining(model, train_data, prune_position_list, args)

def get_cut_point(model, criterion):
    '''
    weight pruning에서 많이 사용.
    전체 파라미터에서 기준비율(criterion) 위치에 있는 파라미터 값을 얻는 코드.
    특정 파라미터 값 하나를 리턴한다.
    '''
    all_param = []
    for i in model.parameters():
        all_param += torch.flatten(i).tolist()
    cut_point = round(len(all_param) * criterion)
    all_param.sort()
    return all_param[cut_point]

def run_pruning(model, threshold):
    '''
    threshold 값보다 작은 값들을 0으로 만든다.
    gradient 업데이트 할 때 0로 만들 포지션 리스트를 리턴으로 반환한다.
    '''
    prune_model = {}
    prune_position_list = []

    for name, tensor in model.state_dict().items():
        tensor[tensor <= threshold] = 0
        prune_model[name] = tensor
        prune_position_list.append(tensor)

    new_state_dict = OrderedDict(prune_model)
    model.load_state_dict(new_state_dict, strict=False)
    return prune_position_list

def retraining(model, train_data, prune_position_list, args):
    '''
    gradient 업데이트 시 pruning 된 포지션은 업데이트 되지 않도록 한다.
    '''
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.learning_rate / args.epochs)

    for epoch in range(args.epochs):
        loss_list = []
        for input, target in train_data:
            model_output = model(input.to(args.device))
            loss = loss_function(model_output, target.to(args.device))
            optim.zero_grad()
            loss.backward()
            for idx, p in enumerate(model.parameters()):
                p.grad = p.grad * prune_position_list[idx]
            optim.step()
            loss_list.append(loss.item())

        loss = sum(loss_list) / len(loss_list)
        scheduler.step(epoch + 1)
        print("retraining {} epoch, loss : {}".format(epoch+1, loss))
