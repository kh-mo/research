'''
step 1. get model
step 2. calculate flop
'''
import argparse

from functions.model_functions import get_model
from functions.utils import cal_flop

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lenet_300_100")
    parser.add_argument("--load_folder_model", type=str, default="None")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--do_training", type=str, default="False")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # step 1
    model = get_model(args)
    print("{} model load complete!!".format(args.model))

    # step 2
    cal_flop(model, args)