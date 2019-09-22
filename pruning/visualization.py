import os
import argparse
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--pruningList", type=float, nargs="+")
    args = parser.parse_args()

    file_list = os.listdir(os.path.join(os.getcwd(), "models"))

    acc_list = []
    pruning_rate_list = []
    for pruning_rate in args.pruningList:
        pruning_rate = '{0:.2f}'.format(pruning_rate)
        for name in file_list:
            file_infor = args.model + "_" + pruning_rate
            if file_infor in name:
                pruning_rate_list.append(pruning_rate)
                acc = float(name.replace(file_infor+"_pruningThreshold_","").split("_acc_")[0])
                acc_list.append(acc)

    plt.plot(pruning_rate_list, acc_list, color='green', marker='o', linestyle='solid')
    plt.xlim(0.4, 0.90)
    plt.ylim(0.8, 1.0)
    plt.xlabel("Parametes Pruned Away")
    plt.ylabel("Accuracy")
    plt.legend(labels = [args.model])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()