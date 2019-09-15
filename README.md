# Research

This repository is used to test different topics what i want to experiment with.

### Topic1 - pruning

### Result

All results are experimented by MNIST.
Non-pruned parameters mean non-zero parameters.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:  
Lenet_300_100 Ref | 97.97% | 266K(266,610) | 
Lenet_300_100 Pruned | 92.37% | 108K(108,026) | 2x 
Lenet_5 Ref | 98.32% | 29K(29,456) | 
Lenet_5 Pruned | 95.05% | 10K(10,395) | 2x 

All results are experimented by Imagenet-2012.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:
Alexnet Ref | OO.OO% | 000K() | 
Alexnet Pruned | OO.OO% | 000K() | 00x 
VGGnet_16 Ref | OO.OO% | 000K() | 
VGGnet_16 Pruned | OO.OO% | 000K() | 00x

## Getting Start
### Get lenet model file
- Get lenet_300_100 or lenet_5 model 
- Possible example : {lenet_300_100-mnist}, {lenet_5-mnist}
```shell
python lenetClassifier.py --model="lenet_300_100" --dataset="mnist"
```

### Get Baseline Result
- Model : lenet, alexnet
- Dataset : mnist, imagenet
- Depending dimension of the input image, there are possible combinations of datasets and models.
- Possible example : {lenet_300_100-mnist}, {lenet_5-mnist}
- to be : {alexnet-imagenet}, {vggnet-imagenet}
```shell
python baseline.py --model="lenet_300_100" --dataset="mnist"
```

### Pruning & Retraining
- Depending dimension of the input image, there are possible combinations of datasets and models.
- possible example : {lenet_300_100-mnist}, {lenet_5-mnist}
- to be : {alexnet-imagenet}, {vggnet-imagenet}
```shell
python PruningRetraining.py --model="lenet_300_100" --dataset="mnist"
```