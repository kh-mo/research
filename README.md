# Research

This repository is used to test different topics what i want to experiment with.

## Topic1 - pruning

### Result

All results are experimented.
Acc() means accuracy(epoch/batch size).
Pruning acc() means accuracy(pruning epochs/retraining epochs/batch size).
Inference time check millisecond time about 1 batch size and average 10 samples.

#### Baseline
*Dataset* | *Model* | *Prune Method* | *Top1 Acc* | Parameters | training time | inference time |
:---: | :---: | :---: | :---: | :---: | :---: | :---: |
MNIST | Lenet-300-100 | none | 98.13%(30/256) | 266,610 | 0 hour 2 minute | 0.4 ms 0.24 variance |
MNIST | Lenet-300-100 | [\[1\] pruning rate 0.5](#reference) | 92.72%(1/30/256) | 133,304(nonzero) | 0 hour 2 minute | 0.8 ms 0.16 variance |
MNIST | Lenet-5 | none | 98.12%(30/256) | 29,456 | 0 hour 2 minute | 1.1 ms 0.09 variance |
MNIST | Lenet-5 | [\[1\] pruning rate 0.5](#reference) | 76.07%(1/30/256) | 14,726(nonzero) | 0 hour 2 minute | 0.8 ms 0.16 variance |

#### Reproduce Paper
For Lenet-300-100
*Layer* | *Weights* | *FLOP* | *Nonzero Weights* |
:---: | :---: | :---: | :---: |
fc1 | 235,500 | 470,400 | 116,247 |
fc2 | 30,100 | 60,000 | 16,580 |
fc3 | 1,010 | 2,000 | 477 |
Total | 266,610 | 532,400 | 133,304 |

For Lenet-5
*Layer* | *Weights* | *FLOP* | *Nonzero Weights* |
:---: | :---: | :---: | :---: |
conv1 | 60 | 73,008 | 44 |
conv2 | 330 | 78,408 | 163 |
fc1 | 27,776 | 55,296 | 13,880 |
fc2 | 1,290 | 2,560 | 639 |
Total | 29,456 | 209,272 | 14,726 |

### Getting Start
#### Get Baseline
- Model : lenet
- Dataset : mnist
```shell
python baseline.py --model=lenet_300_100 --dataset=mnist
```

#### Run pruning method
- Pruning : songhan, fpgm
```shell
python pruning.py --model=lenet_300_100 --dataset=mnist --load_folder_model=alexnet_cifar10_acc_0.8562_epoch_30 --pruning_method=fpgm
```

#### Calculate FLOP
```shell
python calFLOP.py --model=lenet_300_100 --dataset=mnist --load_folder_model=lenet_300_100_mnist_acc_0.9813_epoch_30
```

### Reference
- [1] [Learning both Weights and Connections for Efficient Neural Networks](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf), accepted NIPS 2015
- [2] [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf), accepted CVPR 2019
