# Research

This repository is used to test different topics what i want to experiment with.

## Topic1 - pruning

### Result

All results are experimented. Acc() means accuracy(epoch/batch size). Pruning acc() means accuracy(pruning epochs/retraining epochs/batch size)

*Dataset* | *Model* | *Prune Method* | *Top1 Acc* | Parameters | training time |
:---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | Alexnet | None | 86.3%(30/256) | 61,100,840 | 0 hour 35 minute |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.7](#reference) | 14.97%(1/30/256) | 18,330,251 | 0 hour 37 minute |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.7](#reference) | 18.52%(3/30/256) | 5,703,122 | 1 hour 54 minute |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.7](#reference) | 35.55%(5/30/256) | 2,816,297 | 3 hour 11 minute |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.7](#reference) | 31.9%(7/30/256) | 1,384,413 | 4 hour 24 minute |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.7](#reference) | 32.32%(9/30/256) | 1,062,733 | 5 hour 58 minute |
CIFAR-10 | Resnet-18 | None | 94.35%(30/256) | 11,689,512 | 1 hour 5 minute |
CIFAR-10 | Resnet-18 | [\[1\] pruning rate 0.7](#reference) | 0%(1/30/256) | 000 | 00 hour 00 minute |
CIFAR-10 | Resnet-34 | None | 95.11%(30/256) | 21,797,672 | 1 hour 31 minute |

### Getting Start
#### Get Baseline
- Model : alexnet, resnet
- Dataset : cifar10
```shell
python baseline.py --model=alexnet
```

#### Run pruning method
- Pruning : songhan, fpgm
```shell
python pruning.py --model=alexnet --load_folder_model=alexnet_cifar10_acc_0.8562_epoch_30 --pruning_method=fpgm
```

### Reference
- [1] [Learning both Weights and Connections for Efficient Neural Networks](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf), accepted NIPS 2015
- [2] [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf), accepted CVPR 2019
