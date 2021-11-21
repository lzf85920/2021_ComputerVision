# Conditional image synthesis and Feature Disentanglement

###### tags: `Deep Learning for Computer Vision`

In this Task, I applied AC-GAN to implement conditional image generation.

![](https://i.imgur.com/HXTwoQH.png)

## AC-GAN 


![](https://i.imgur.com/NUUBFmw.png)

### Generator

``` python
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size):
      self.channel = 3
      super(Generator, self).__init__()

      self.label_emb = nn.Embedding(n_classes, latent_dim)

      self.init_size = img_size // 4  # Initial size before upsampling
      self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

      self.conv_blocks = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(128, 128, 3, stride=1, padding=1),
          nn.BatchNorm2d(128, 0.8),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(128, 64, 3, stride=1, padding=1),
          nn.BatchNorm2d(64, 0.8),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(64, self.channel, 3, stride=1, padding=1),
          nn.Tanh(),
      )

    def forward(self, noise, labels):
      gen_input = torch.mul(self.label_emb(labels), noise)
      out = self.l1(gen_input)
      out = out.view(out.shape[0], 128, self.init_size, self.init_size)
      img = self.conv_blocks(out)
      return img

```



### Discriminator
``` python
# https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations

class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size):
        super(Discriminator, self).__init__()
        self.channel = 3
        self.n_classes = n_classes
        self.init_size = img_size
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channel, 16, bn=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.init_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
```

### Hyperparameters :
* Batch size : 50
* Number of epochs : 100
* Image size : 32*32
* Learning rate : 0.0002
* latent vector : 100*1
* Optimizer : Adam(betas=(0.5, 0.999))

### Training detial :
When we are training ACGAN, we must input label and random noise to the model together.

First, we convert the label into one hot vector and merge it with random noise
``` python
n_classes = Variable(LongTensor(np.random.randint(0, num_classes, batch_size)))
self.label_emb = nn.Embedding(n_classes, latent_dim)
```

Next, we input the fake image into the discriminator, and we will get two loss
``` python
validity, pred_label = discriminator(gen_imgs)
g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
g_loss.backward()
```

### Model Ensemble :
I selected ten models from the last 10 epochs and averaged all the parameters.

``` python
import torch as t

model1 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_90.pth')
model2 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_99.pth')
model3 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_98.pth')
model4 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_97.pth')
model5 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_96.pth')
model6 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_95.pth')
model7 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_94.pth')
model8 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_93.pth')
model9 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_92.pth')
model10 = t.load('/content/drive/MyDrive/HW2/model/acgan/acgan_91.pth')

for key, value in model1.items():
      model1[key] = (value + model2[key] + model3[key] + model4[key] + model5[key] + model6[key] + model7[key] + model8[key] + model9[key] + model10[key]) / 10

generator = Generator(nz, num_classes, image_size).cuda()
generator.load_state_dict(model1)

t.save(generator.state_dict(),  '/content/drive/MyDrive/HW2/model/acgan/acgan_ensemble.pth')
```

### Example Results
![](https://i.imgur.com/HXTwoQH.png)


### Accuracy
We load a pre-trained classifier to predict the category of images generated from ACGAN


``` python
Accuracy: 0.991
```
