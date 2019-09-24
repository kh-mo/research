import os
import argparse
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--pruningList", type=float, nargs="+")
    args = parser.parse_args()

    file_list = os.listdir(os.path.join(os.getcwd(), "models"))

    acc_list = [[] for i in range(len(args.models))]
    pruning_rate_list = [[] for i in range(len(args.models))]
    for model_idx, model in enumerate(args.models):
        for pruning_rate in args.pruningList:
            pruning_rate = '{0:.2f}'.format(pruning_rate)
            file_infor = model + "_" + pruning_rate
            for name in file_list:
                if file_infor in name:
                    pruning_rate_list[model_idx].append(pruning_rate)
                    acc = float(name.replace(file_infor+"_pruningThreshold_","").split("_acc_")[0])
                    acc_list[model_idx].append(acc)

    plt.xlim(0.4, 0.90)
    plt.ylim(0.8, 1.0)
    for i in range(len(acc_list)):
        plt.plot(pruning_rate_list[i], acc_list[i], marker='o', linestyle="solid")
    plt.xlabel("Parametes Pruned Away")
    plt.ylabel("Accuracy")
    plt.legend(labels = args.models)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()