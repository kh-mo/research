'''
pretrained model : https://pytorch.org/docs/stable/torchvision/models.html
dataset : https://pytorch.org/docs/stable/torchvision/datasets.html

step 1. get model, dataset
step 2. training(option)
step 3. evaluate
step 4. save model
step 5. check time
'''
import time
import argparse

import torch

from functions.model_functions import get_model
from functions.dataset_functions import get_dataset
from functions.utils import evaluating, training, saving

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--load_folder_model", type=str, default="None")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--do_training", type=str, default="True")
    parser.add_argument("--learning_rate", type=int, default=0.001)
    parser.add_argument("--accuracy", type=float, default=0.)
    parser.add_argument("--param_count", type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # step 1
    model = get_model(args)
    print("{} model load complete!!".format(args.model))

    train_data, test_data = get_dataset(args)
    print("{} dataset load complete!!".format(args.dataset))

    # step 2
    train_start_time = time.time()
    if model.training:
        print("start training")
        training(model, train_data, args)
        print("training done.")
    train_end_time = time.time()

    # step 3
    inference_start_time = time.time()
    print("use {} for evaluating".format(args.device))
    args.accuracy, args.param_count = evaluating(model, test_data, args)
    inference_end_time = time.time()

    # step 4
    saving(model, args)
    print("Complete model saving")

    # step 5
    total_train_time = train_end_time - train_start_time
    total_inference_time = inference_end_time - inference_start_time
    print("train time : {} hour {} minite".format(int(total_train_time/3600), int((total_train_time%3600)/60)))
    print("inference time : {} hour {} minite".format(int(total_inference_time/3600), int((total_inference_time%3600)/60)))
