# Research

This repository is used to test different topics what i want to experiment with.

### Topic1 - pruning

### Result

All results are experimented by MNIST.
Non-pruned parameters mean non-zero parameters.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:  
Lenet_300_100 Ref | 97.97% | 266K(266,610) | 
Lenet_300_100 Pruned | 92.46% | 15K(15,023) | 17x 
Lenet_5 Ref | 98.32% | 29K(29,456) | 
Lenet_5 Pruned | 95.15% | 1K(1,603) | 18x 

Accuracy Graph for threshold determined by percentage of initial network weights.

<img src = 'saved_image/graph_image1.JPG'>

All results are experimented by Imagenet-2012.

*Model* | *Accuracy* | *Parameters* | *Compression Rate*
:---: | :---: | :---: | :---:
Alexnet Ref | 54.85% | 61,100K(61,100,840) |
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
python PruningRetraining.py --model="lenet_300_100" --dataset="mnist" --pruningThresholds 0.5 0.6 0.7 0.8
```

### Visualizing
```shell
python visualization.py --models lenet_300_100 lenet_5 --pruningList 0.4 0.5 0.6 0.7 0.8 0.9
```