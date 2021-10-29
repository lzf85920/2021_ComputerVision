# Image semantic segmentation

###### tags: `Deep Learning for Computer Vision`


## Image semantic segmentation

In this Task, I applied **VGG16-FCN32s** and **DeepLabV3-ResNet101** to implement semantic segmentation.

<img src="https://i.imgur.com/vghP24D.jpg" width="200"/> <img src="https://i.imgur.com/EDA3XqT.png" width="200"/> <img src="https://i.imgur.com/QlzwrLp.png" width="200"/>

Real Image > Ground Truth > Prediction 

### VGG16-FCN32s [1]

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/hhf7zj2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">VGG16</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/gZ8gFb3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;margin-bottom: 10%;
    display: inline-block;
    color: #999;
    padding: 2px;">Fully Convolution Networks</div>
</center>


``` python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models 
from torchvision.models.vgg import VGG

class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        score = self.relu(self.conv6(x5))
        score = self.relu(self.conv7(score))              # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(score))            # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        
        score = self.relu(self.conv6(x5))
        score = self.relu(self.conv7(score))
        score = self.relu(self.deconv1(score))            # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        
        score = self.relu(self.conv6(x5))                 # size=(N, 512, x.H/32, x.W/32)
        out_conv7 = self.relu(self.conv7(score))          # size=(N, 512, x.H/32, x.W/32)
        out_conv7 = self.relu(self.deconv1(out_conv7))    # size=(N, 512, x.H/16, x.W/16)
        four_conv7 = self.relu(self.deconv2(out_conv7))   # size=(N, 256, x.H/8, x.W/8)
        
        two_pool4 = self.relu(self.deconv2(x4))           # size=(N, 256, x.H/8, x.W/8)
        
        score = self.bn2(four_conv7 + two_pool4 + x3)     # element-wise add, size=(N, 256, x.H/8, x.W/8)
        
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

#### Data Augmentation :
We have 2000 512x512 pixel satellite images for training but it was not enough. So I use the pytorch RandomCrop package to randomly crop each image 6 times plus HorizontalFlip, VerticalFlip and randomly rotate image during training. Finally, we get 20,000 256x256 pixel images.

``` python
# Randomly rotate image 
if self.train:
  randomly = np.random.random()
  if randomly > 0.5:
    angle = [20, 30, 35, 40, 45, 55, 65, 75]
    select_angle = np.random.choice(angle)
    img = transforms.functional.rotate(img, int(select_angle))
    label = transforms.functional.rotate(label, int(select_angle))
  else:
    pass
else:
  pass
```

Note that : when we crop the training data, the corresponding label must also be cropped in the same area.

#### Hyperparameters :
* Batch size : 8
* Number of epochs : 30
* Learning rate : 0.0001
* Lr_scheduler : 0.5 * lr every 10 epoch


#### Model Ensemble :
I selected three models with the lowest loss from the training process and averaged all the parameters.
``` python
model1 = t.load('/content/drive/MyDrive/HW1/model/vgg16-fcn32_7.pth')
model2 = t.load('/content/drive/MyDrive/HW1/model/vgg16-fcn32_8.pth')
model3 = t.load('/content/drive/MyDrive/HW1/model/vgg16-fcn32_9.pth')

for key, value in model1.items():
      model1[key] = (value + model2[key] + model3[key]) / 3

vgg_model = VGGNet(requires_grad=True) 
ensemble = FCN32s(pretrained_net=vgg_model,n_class=7)
ensemble.load_state_dict(model1)

t.save(ensemble.state_dict(),  '/content/drive/MyDrive/HW1/model/vgg16-fcn32_ensemble.pth')
```

#### Validation Result : 
``` python
mean_iou: 0.695547
```


### DeepLabV3-ResNet101 [2]

The DeepLab model addresses this challenge by using Atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) modules. This architecture has evolved over several generations:

DeepLabV1 : Uses Atrous Convolution and Fully Connected Conditional Random Field (CRF) to control the resolution at which image features are computed.

DeepLabV2 : Uses Atrous Spatial Pyramid Pooling (ASPP) to consider objects at different scales and segment with much improved accuracy.

DeepLabV3 : Apart from using Atrous Convolution, DeepLabV3 uses an improved ASPP module by including batch normalization and image-level features. It gets rid of CRF (Conditional Random Field) as used in V1 and V2.


#### DeepLabV3 Model Architecture
* Features are extracted from the backbone network (VGG, DenseNet, ResNet).
* To control the size of the feature map, atrous convolution is used in the last few blocks of the backbone.
* On top of extracted features from the backbone, an ASPP network is added to classify each pixel corresponding to their classes.
* The output from the ASPP network is passed through a 1 x 1 convolution to get the actual size of the image which will be the final segmented mask for the image.


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/HYDRbJn.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">DeepLabV3 Model Architecture</div>
</center>

#### Validation Result : 
``` python
mean_iou: 0.753360
```

### Training Process

![](https://i.imgur.com/iifpeW1.png)


## References
[1] https://blog.csdn.net/weixin_43143670/article/details/104791946

[2] https://developers.arcgis.com/python/guide/how-deeplabv3-works/
