# Research

This repository is used to test different topics what i want to experiment with.

### Topic1 - pruning

### Result

All results are experimented by MNIST.

*Model* | *Accuracy* 
:---: | :---: 
Lenet-300-100 ref | 97.95%
Lenet-300-100 pru | 00.00% 

All results are experimented by Imagenet-2012.

*Model* | *Accuracy* 
:---: | :---: 
Alexnet | OO.OO% 

## Getting Start
### Get Baseline Result
- model : lenet, alexnet
- dataset : mnist, imagenet
- Depending dimension of the input image, there are possible combinations of datasets and models.
- example : lenet-mnist, alexnet-imagenet
```shell
python baseline.py --model="lenet" --dataset="mnist"
```