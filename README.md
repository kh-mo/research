# Research

This repository is used to test different topics what i want to experiment with.

### Topic1 - pruning

### Result

All results are experimented by MNIST.

*Model* | *Accuracy* 
:---: | :---: 
Lenet_300_100 Ref | 97.95%
Lenet_300_100 Pruned | 00.00% 
Lenet_5 Ref | 00.00%
Lenet_5 Pruned | 00.00% 

All results are experimented by Imagenet-2012.

*Model* | *Accuracy* 
:---: | :---: 
Alexnet Ref | OO.OO%
Alexnet Pruned | OO.OO% 
VGGnet_16 Ref | OO.OO%
VGGnet_16 Pruned | OO.OO%

## Getting Start
### Get Baseline Result
- model : lenet, alexnet
- dataset : mnist, imagenet
- Depending dimension of the input image, there are possible combinations of datasets and models.
- possible example : {lenet_300_100-mnist}, {lenet_5-mnist}, {alexnet-imagenet}, {vggnet-imagenet}
```shell
python baseline.py --model="lenet_300_100" --dataset="mnist"
```