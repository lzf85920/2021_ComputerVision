# Image Classification with Vision Transformer

###### tags: `Deep Learning for Computer Vision`

In this Task, I applied **Vision Transformer** to implement image classification.

<img src="https://i.imgur.com/FQ5dAyY.jpg" width="450"/>

## Vision Transformer

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/ME1xCAK.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">ViT</div>
</center>

<br>

``` python
from pytorch_pretrained_vit import ViT

model = ViT('B_16_imagenet1k', pretrained=True)
n_classes = 37
model.fc = nn.Linear(in_features=768, out_features=n_classes, bias=True)
```

### Hyperparameters :
* Batch size : 16
* Number of epochs : 30
* Image size : 384*384
* Learning rate : 0.002
* Learning rate scheduler : 0.7 * lr every 5 epoch
* Optimizer : SGD(momentum=0.9)


### Model Ensemble :
I selected three hightest accuracy models during the training then averaged all the parameters.
``` python
import torch as t

model1 = t.load('/content/drive/MyDrive/HW3/model/vit_v2_11.pth')
model2 = t.load('/content/drive/MyDrive/HW3/model/vit_v2_12.pth')
model3 = t.load('/content/drive/MyDrive/HW3/model/vit_v2_28.pth')

for key, value in model1.items():
      model1[key] = (value + model2[key] + model3[key]) / 3

from pytorch_pretrained_vit import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViT('B_16_imagenet1k', pretrained=True)
n_classes = 37

ensemble = ViT('B_16_imagenet1k', pretrained=True)
ensemble.fc = nn.Linear(in_features=768, out_features=n_classes, bias=True)
ensemble = ensemble.to(device)
ensemble.load_state_dict(model1)

t.save(ensemble.state_dict(),  '/content/drive/MyDrive/HW3/model/vit_v2_ensemble.pth')
  
```

### Validation Accuracy

| model | val_acc |
| -------- | -------- |
| B_16_imagenet1k  | 0.9441 |
| B_32_imagenet1k  | 0.9322 |
| L_16_imagenet1k  | 0.9380 |

After training three different models, I found that a simple model will get better performance.

### Visualize position embeddings
The point at $(i, j)$ indicates the similarity between the $i$-th position and the $j$-th position. we can only observe that embedding vectors are similar to the positions nearby but have no explainable patterns in long-term relations.

<img src="https://i.imgur.com/81IgDdx.png" width="400"/>

### Visualize Attention Map
Attention map is a heatmap which tell you where is the model focus on. When you flatten query key matrix, the high attention rate represent which patch is more important.

``` python
# https://github.com/lukemelas/PyTorch-Pretrained-ViT/issues/19
def forward(self, x):
    b, c, fh, fw = x.shape
    x = self.patch_embedding(x)  # b,d,gh,gw
    x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
    if hasattr(self, 'class_token'):
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
    if hasattr(self, 'positional_embedding'): 
        x = self.positional_embedding(x)  # b,gh*gw+1,d 
    x,atten_scores = self.transformer(x)  # b,gh*gw+1,d
    att_mat = torch.stack(atten_scores).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    # print("att_mat",att_mat.shape)
    if hasattr(self, 'pre_logits'):
        x = self.pre_logits(x)
        x = torch.tanh(x)
    if hasattr(self, 'fc'):
        x = self.norm(x)[:, 0]  # b,d
        x = self.fc(x)  # b,num_classes
    return x,att_mat
```

<img src="https://i.imgur.com/FQ5dAyY.jpg" width="450"/>

From the above figure, we can see our model focuses on which patch. Although there are some noises, the model still pays attention to the patch where the object should be recognized.
