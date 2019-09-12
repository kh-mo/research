# Research

This repository is used to test different topics what i want to experiment with.

### Topic1 - pruning

### Result

All results are experimented by MNIST.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:  
Lenet_300_100 Ref | 97.97% | 000K | 00x
Lenet_300_100 Pruned | 92.37% | 000K | 00x 
Lenet_5 Ref | 00.00% | 000K | 00x
Lenet_5 Pruned | 00.00% | 000K | 00x 

All results are experimented by Imagenet-2012.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:
Alexnet Ref | OO.OO% | 000K | 00x
Alexnet Pruned | OO.OO% | 000K | 00x 
VGGnet_16 Ref | OO.OO% | 000K | 00x
VGGnet_16 Pruned | OO.OO% | 000K | 00x

## Getting Start
### Get lenet model file
- Get lenet_300_100 or lenet_5 model 
- Possible example : {lenet_300_100-mnist}
- to be : {lenet_5-mnist}
```shell
python lenetClassifier.py --model="lenet_300_100" --dataset="mnist"
```

### Get Baseline Result
- Model : lenet, alexnet
- Dataset : mnist, imagenet
- Depending dimension of the input image, there are possible combinations of datasets and models.
- Possible example : {lenet_300_100-mnist}
- to be : {lenet_5-mnist}, {alexnet-imagenet}, {vggnet-imagenet}
```shell
python baseline.py --model="lenet_300_100" --dataset="mnist"
```

### Pruning & Retraining
- Depending dimension of the input image, there are possible combinations of datasets and models.
- possible example : {lenet_300_100-mnist}
- to be : {lenet_5-mnist}, {alexnet-imagenet}, {vggnet-imagenet}
```shell
python PruningRetraining.py --model="lenet_300_100" --dataset="mnist"
```