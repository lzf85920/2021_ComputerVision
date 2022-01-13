# Few-Shot Learning

###### tags: `Deep Learning for Computer Vision`

## Prototypical Network

In this Task, I applied **Prototypical Network** to perform 5-way 1-shot classification.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/Mfcd8H4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">prototypical network</div>
</center>

### Feature Extractor :

``` python
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, para=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
      x = self.encoder(x)
      return x.view(x.size(0), -1)

```

### Implementation details
* Meta-train : 20way-1shot
* Meta-test : 5way-1shot

#### Hyperparameters
* Number of epochs : 100
* Learning rate : 0.001
* Lr_scheduler : 0.5 * lr every 20 step

During the meta-train. We randomly select n categories and k samples from the data set, and then use the model to calculate the prototype (A total of n prototypes).

During the meta-test. We also use the feature extractor to calculate the prototype of the query and calculate the distance (euclidean distance) between the two prototypes.


### Validation Result : 
``` python
Accuracy: 46.34 +- 0.82 %
```


### Different distance function

* Meta-train : 5way-1shot
* Meta-test : 5way-1shot

#### Euclidean
``` python 
Accuracy: 40.77 +- 0.77 %
```
#### Cosin similarity
``` python 
Accuracy: 24.77 +- 0.45 %
```

#### Parametric function

``` python
class Parametric(nn.Module):

    def __init__(self, query, train_way):
        super().__init__()
        self.para_dis = nn.Sequential(
          nn.Linear(query*train_way*train_way, 64),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, query*train_way*train_way)
          )

    def forward(self, x):
      x = self.para_dis(x)
      return x
```

``` python 
Accuracy: 20.14 +- 0.37 %
```
### Different K shot settings

| 5way-1shot | 5way-5shot | 5way-10shot |
| -------- | -------- | -------- |
| 0.4067 | 0.6236  | 0.6900  |
