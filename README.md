# LdsConv
LdsConv: Learned Depthwise Separable Convolutions by Group Pruning

# Requirments
pytorch >= 1.1

# Get Start
Prepare your dataset
Then just python train.py or python imagenet_train_densenet.py
# Results
|Model|Error(top-1)|GFLOPs|Params(M)|
|----|----|----|----|
|Lds-ResNet50 (k=2)|22.9|2.71|16.8|
|Lds-ResNet-extreme|23.4|2.28|14.3|
