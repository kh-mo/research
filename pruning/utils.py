import torch

def evaluate(model, test_data, args):
    pred = []
    label = []
    for test_img, test_label in test_data:
        pred += torch.argmax(model(test_img.to(args.device)),dim=1).tolist()
        label += test_label.tolist()
    total_acc = sum([1 if pred[i] == label[i] else 0 for i in range(len(pred))]) / len(pred)
    print(total_acc)

def visualization():
    return