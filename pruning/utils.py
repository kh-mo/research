import torch
import numpy as np

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

def visualization():
    return