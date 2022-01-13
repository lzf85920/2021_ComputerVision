# Bootstrap Your Own Latent 

###### tags: `Deep Learning for Computer Vision`

In this Task, we pre-train own ResNet50 backbone on Mini-ImageNet via the recently self-supervised learning methods (BYOL).

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/3KW5tXJ.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">prototypical network</div>
</center>

### Implementation details

During the pre-training, I train a total of 80 epochs, batch size is 32 and image size is 128*128.

The pre-train dataset consists of 48,000 84x84 RGB images in 80 classes. When I completed the self-monitoring training, I spent a total of 20 hours


## Downstream Classification Task

### Implementation details

When I completed the self-monitoring training. I use pre-training **Resnet50** backbone and add a fully connected layer to perform classification.

#### Fully Connected Layer

``` python
(0): Linear(in_features=1000, out_features=512, bias=True)
(1): ReLU()
(2): Dropout(p=0.5, inplace=False)
(3): Linear(in_features=512, out_features=128, bias=True)
(4): Linear(in_features=128, out_features=65, bias=True)
```

### Cross-Comparison Experiment


We can find that if there is no fine tune feature extractor, the performance of the model will be poor (fix resnet50).
Furthermore, the backbone training of the supervised method is worse than the self-supervised method. I think supervised method cannot learn to extract robust feature representation.

| Setting |        Pre-training (Mini-ImageNet)        |    Fine-tuning (Office-Home dataset)     | Classification accuracy on valid set (Office-Home dataset) |
|:-------:|:------------------------------------------:|:----------------------------------------:|:----------------------------------------------------------:|
|    A    |                     -                      | Train full model (backbone + classifier) |                           0.1554                           |
|    B    | w/ label (TAs have provided this backbone) | Train full model (backbone + classifier) |                           0.1795                           |
|    C    | w/o label (Your SSL pre-trained backbone)  | Train full model (backbone + classifier) |                           **0.5385**                           |
|    D    | w/ label (TAs have provided this backbone) | Fix the backbone. Train classifier only  |                           0.1074                           |
|    E    | w/o label (Your SSL pre-trained backbone)  | Fix the backbone. Train classifier only  |                           0.2324                           |



