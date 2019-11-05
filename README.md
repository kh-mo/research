# Research

This repository is used to test different topics what i want to experiment with.

## Topic1 - pruning

### Result

All results are experimented.
Acc() means accuracy(epoch/batch size).
Pruning acc() means accuracy(pruning epochs/retraining epochs/batch size).
Inference time check millisecond time about 1 batch size and average 10 samples.

#### Baseline
*Dataset* | *Model* | *Top1 Acc* | Parameters | training time | inference time |
:---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | Lenet-300-100 | 0%(30/256) | 0 | 00 hour 0 minute | 0 ms 0 variance |

#### Reproduce Paper
*Dataset* | *Model* | *Prune Method* | *Top1 Acc* | Parameters | training time | inference time |
:---: | :---: | :---: | :---: | :---: | :---: | :---: |

*Dataset* | *Model* | *Prune Method* | *Top1 Acc* | Parameters | training time | inference time |
:---: | :---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | Alexnet | None | 86.3%(30/256) | 61,100,840 | 00 hour 35 minute | 2.4 ms 0.24 variance |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.5](#reference) | 11.63%(1/30/256) | 30,548,391 | 00 hour 40 minute | 1.6 ms 23.04 variance |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.5](#reference) | 00.00%(3/30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.5](#reference) | 00.00%(5/30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.5](#reference) | 00.00%(7/30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Alexnet | [\[1\] pruning rate 0.5](#reference) | 00.00%(9/30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Resnet-18 | None | 00.00%(30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Resnet-18 | [\[1\] pruning rate 0.5](#reference) | 00.00%(1/30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |
CIFAR-10 | Resnet-34 | None | 00.00%(30/256) | 000 | 00 hour 00 minute | 00 ms 00 variance |

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
