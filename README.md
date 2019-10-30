# Research

This repository is used to test different topics what i want to experiment with.

## Topic1 - pruning

### Result

All results are experimented. Acc() means accuracy(epoch/batch size).

*Dataset* | *Model* | *Prune Method* | *Top1 Acc* | Parameters |
:---: | :---: | :---: | :---: | :---: |
CIFAR-10 | Alexnet | None | 85.62%(30/256) | 61,100,840 |
CIFAR-10 | Alexnet | [\[1\]](### Reference) | 00.00%(0/00) | 000 |
CIFAR-10 | Resnet-18 | None | 94.21%(30/256) | 11,689,512 |
CIFAR-10 | Resnet-34 | None | 94.62%(30/256) | 21,797,672 |

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
python pruning.py --model=alexnet --pruning=fpgm
```

### Reference
- [1] [Learning both Weights and Connections for Efficient Neural Networks](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf), accepted NIPS 2015
- [2] [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf), accepted CVPR 2019
