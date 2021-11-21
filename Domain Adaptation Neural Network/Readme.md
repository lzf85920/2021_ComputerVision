# Domain Adaptation Neural Network

###### tags: `Deep Learning for Computer Vision`

In this task,  I applied DANN to implement domain adaptation.

![](https://i.imgur.com/fYyi5Q5.jpg)


## scenario
* source domain : SVHN
* Target domain : MNIST-M

### training on source domain only (Lower bound)
Use source images and labels in the training folder for training, target images and labels in the testing folder to compute the accuracy

``` python
Avg Accuracy = 0.4925
```

### training on source and target domain (domain adaptation)
Use source images and labels in the training folder + target images in the training folder for training, target images and labels in the testing folder to compute the accuracy

``` python
Avg Accuracy = 0.5683
```

### training on target domain only (Upper bound)
Use target images and labels in the training folder for training, target images and labels in the testing folder to compute the accuracy

``` python
Avg Accuracy = 0.9798
```

## Result Matrix


| | MNIST-M → USPS | SVHN → MNIST-M | USPS → SVHN |
| -------- | -------- | -------- | -------- |
| Trained on source| 0.7429 | 0.4925 | 0.1526 |
| Adaptation | 0.7693 | 0.5683 | 0.3126 |
| Trained on target| 0.9626 | 0.9788 | 0.9155 |


## Visualization

![](https://i.imgur.com/MdtMGBW.png)
![](https://i.imgur.com/oFi3C3S.png)
![](https://i.imgur.com/4ccRSkc.png)

## Remark
* In addition to caculate classification loss of source , the loss function also adds domain loss of source and target. (Ie, loss = source_loss_class + source_loss_domain + target_loss_domain)
* During the training process, we expect that the accuracy of the domain classifier will get closer and closer to 0.5, which means that the features of the source and target domains have been mixed together.


## Improved UDA model 
### scenario 1
* source domain : SVHN
* Target domain : MNIST-M

#### Original

``` python
# https://github.com/wogong/pytorch-dann
SVHNmodel(
  (feature): Sequential(
    (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (6): ReLU(inplace=True)
    (7): Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1))
  )
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=1024, out_features=256, bias=True)
    (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=256, out_features=10, bias=True)
  )
  (discriminator): Sequential(
    (0): Linear(in_features=128, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=1024, out_features=256, bias=True)
    (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=256, out_features=2, bias=True)
  )
)
```


#### Improved
I add batchnorm and relue to the featurer extractor
``` python
Sequential(
  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (4): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
  (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Dropout2d(p=0.5, inplace=False)
  (7): ReLU(inplace=True)
  (8): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (9): ReLU(inplace=True)
  (10): Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1))
)
```

### scenario 1
* source domain : MNIST-M
* Target domain : USPS

#### Original

``` python
# https://github.com/wogong/pytorch-dann
MNISTMmodel(
  (feature): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 48, kernel_size=(5, 5), stride=(1, 1))
    (5): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout2d(p=0.8, inplace=False)
    (7): ReLU(inplace=True)
    (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=768, out_features=300, bias=True)
    (1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=300, out_features=100, bias=True)
    (4): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=100, out_features=10, bias=True)
  )
  (discriminator): Sequential(
    (0): Linear(in_features=768, out_features=300, bias=True)
    (1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=300, out_features=2, bias=True)
  )
)
```


#### Improved

I reduce the number of layers of the classifier and discriminator, and then add batchnorm and relue to the featurer extractor

``` python
MNISTMmodel(
  (feature): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 48, kernel_size=(5, 5), stride=(1, 1))
    (5): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout2d(p=0.5, inplace=False)
    (7): ReLU(inplace=True)
    (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=768, out_features=100, bias=True)
    (1): Linear(in_features=100, out_features=10, bias=True)
  )
  (discriminator): Sequential(
    (0): Linear(in_features=768, out_features=100, bias=True)
    (1): Linear(in_features=100, out_features=2, bias=True)
  )
)
```


### scenario 3
* source domain : USPS
* Target domain : SVHN

#### Original

``` python
# https://github.com/wogong/pytorch-dann
USPSMmodel(
  (feature): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 24, kernel_size=(5, 5), stride=(1, 1))
    (5): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout2d(p=0.8, inplace=False)
    (7): ReLU(inplace=True)
    (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=384, out_features=24, bias=True)
    (1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=24, out_features=10, bias=True)
  )
  (discriminator): Sequential(
    (0): Linear(in_features=384, out_features=24, bias=True)
    (1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=24, out_features=2, bias=True)
  )
)
```


#### Improved

1. Data Preprocess
``` python
transforms.Compose([
    transforms.CenterCrop((28, 20)),
    transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=3),
])
```
<img src="https://i.imgur.com/OqlsJND.png" width="300"/>

2. Model Architecture
``` python
dann.feature[4] = nn.Conv2d(32, 24, kernel_size=(5, 5), stride=(1, 1))
dann.feature[5] = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
dann.feature[6] = nn.Dropout2d(p=0.5, inplace=False)

dann.classifier[0] = nn.Linear(in_features=24*4*4, out_features=24, bias=True)
dann.classifier[1] = nn.BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
dann.classifier[2] = nn.Sigmoid()
dann.classifier[3] = nn.Linear(in_features=24, out_features=10, bias=True)

dann.discriminator[0] = nn.Linear(in_features=24*4*4, out_features=24, bias=True)
dann.discriminator[1] = nn.BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
dann.discriminator[2] = nn.Sigmoid()
dann.discriminator[3] = nn.Linear(in_features=24, out_features=2, bias=True)
```

### Improved Result

| | MNIST-M → USPS | SVHN → MNIST-M | USPS → SVHN |
| -------- | -------- | -------- | -------- |
| Original model | 0.7693 | 0.5683 | 0.3126 |
| Improved model | 0.7972 | 0.5849 | 0.3726 |



### Remark
* We can transform source domain data into similar to target domain during data preprocessing.
* The model architecture with some few layer and number of neural will be better performance, when the source domain data (USPS, rgb=1) is more colorless than target domain data (SVHN, rgb=3).
