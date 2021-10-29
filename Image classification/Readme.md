# Image classification

###### tags: `Deep Learning for Computer Vision`

## Image classification

In this Task, I applied Inception-V3 to implement image classification.

#### Training Tips :
* Auto Augmetation [1]
* Random horizontal flip : p = 0.5
* Normalize : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
* Optimizer : Stochastic Gradient Decent (SGD)
* Learning rate scheduling : Reduce LR on Plateau
* Clipping Gradient : Max norm = 10



#### Hyperparameters :
* Batch size : 32
* Number of epochs : 20
* Learning rate : 0.0015
* Lr_scheduler factor : 0.5
* Lr_scheduler threshold : 0.0001
* Lr_scheduler patience : 10


### Inception-V3 [2]:

``` python
torchvision.models.inception_v3(aux_logits=False, pretrained=True)
```

<img src="https://i.imgur.com/4AjMo50.png" width="400"/>
<img src="https://i.imgur.com/z0nl7FG.png" width="320"/>
<img src="https://i.imgur.com/obJFBdx.png" width="320"/>
<img src="https://i.imgur.com/C4uUnBz.png" width="320"/>
<img src="https://i.imgur.com/PWUZm8r.png" width="320"/>



### Validation
``` python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.inception_v3(aux_logits=False, pretrained=True)
model.load_state_dict(torch.load('/content/drive/MyDrive/HW1/model/inv3.pth'))
model.to(device)
out = []

# return list siez of (lenght = N/batchsize), (batchsize x layer output dimension) on every list item.  
# Get the output from specific layer.
def hook(module, input, output):
  out.append(output) 

# Add this line that you can automatically save the specific output.
model.avgpool.register_forward_hook(hook) 
model.eval()

predictions = []
valid_accs = []
y_label = []

for batch in valid_loader:
   
  imgs, labels = batch
  y_label.append(labels)

  # Using torch.no_grad() accelerates the forward process.
  with torch.no_grad():
    logits = model(imgs.to(device)) # The shape of logits is batchsize x layer output dimension.

  # Take the class with greatest logit as prediction and record it.
  predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
  acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
  valid_accs.append(acc)

valid_acc = sum(valid_accs) / len(valid_accs)
print("val_acc = ", str(round(valid_acc.item(), 4)))
```

``` python
val_acc = 0.8655
```


### Visualization (t-SNE)
``` python
outputs = []
for batch in range(len(out)):
  for i in np.array(out[batch].to('cpu')):
    outputs.append(i.reshape(1, -1)[0].reshape(1, -1)[0]) # reshape to 1 x dimension
# we got N x fearure number dimension after flatten the output. (batchsize shape)

outputs = np.array(outputs)
print('The shape of outputs:', outputs.shape)

y = []
for i in y_label:
  for j in i:
    y.append(j.item())
y = np.array(y)

print('The shape of label:', y.shape)
```
``` python
The shape of output: (2500, 2048)
The shape of label: (2500, 1)
```

``` python
# https://mortis.tech/2019/11/program_note/664/
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

# # t-SNE
X = outputs
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1, n_iter=2000).fit_transform(X)

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis', figsize=(20,20))
```
![](https://i.imgur.com/CJokRxE.png)

We take output features of the second last layer and apply **t-Distributed Stochastic Neighbor Embedding** (t-SNE) Embedding algorithm [3] to visualize. 

According to t-SNE results, we clustering most of the data well. But there are still some data that can't be clustered.


---




## References
[1] https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
[2] https://arxiv.org/pdf/1512.00567.pdf
[3] https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
