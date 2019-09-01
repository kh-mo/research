import os
os.chdir(os.path.join(os.getcwd(), "pruning"))

import torch
import argparse
from models import get_model
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, device)
    prune_model = {}
    threshold = 0.0

    for name, tensor in model.state_dict().items():
        prune_position = torch.clamp(tensor, min=threshold)
        prune_position[prune_position > threshold] = 1
        prune_model[name] = tensor * prune_position

    new_state_dict = OrderedDict(prune_model)
    model.load_state_dict(new_state_dict, strict=False)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "models/pruned_{}".format(args.model)))

