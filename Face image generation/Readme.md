# Face image generation

###### tags: `Deep Learning for Computer Vision`

In this Task, I applied **DC-GAN** to implement Face image generation.

![](https://i.imgur.com/6TbdWMe.png)

## DC-GAN

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);margin: 2%;" 
    src="https://i.imgur.com/sUsGjcs.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">DCGAN</div>
</center>

### Generator
``` python
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```
### Discriminator

``` python
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

### Hyperparameters :
* Batch size : 64
* Number of epochs : 100
* Image size : 64*64
* Learning rate : 0.0002
* latent vector : 100*1
* Learning rate scheduler : 0.8 * lr every 10 epoch
* Optimizer : Adam(betas=(0.5, 0.999))


### Model Ensemble :
I selected ten models from the last 10 epochs and averaged all the parameters.
``` python
import torch as t

model1 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_100.pth')
model2 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_99.pth')
model3 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_98.pth')
model4 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_97.pth')
model5 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_96.pth')
model6 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_95.pth')
model7 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_94.pth')
model8 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_93.pth')
model9 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_92.pth')
model10 = t.load('/content/drive/MyDrive/HW2/model/dcgan/dcgan_91.pth')

for key, value in model1.items():
      model1[key] = (value + model2[key] + model3[key] + model4[key] + model5[key] + model6[key] + model7[key] + model8[key] + model9[key] + model10[key]) / 10

ensemble = Generator(ngpu).to(device)
ensemble.load_state_dict(model1)

t.save(ensemble.state_dict(),  '/content/drive/MyDrive/HW2/model/dcgan/dcgan_ensemble.pth')
  
```

### Example Results
![](https://i.imgur.com/6TbdWMe.png)


### Fréchet inception distance (FID)
``` python
# https://github.com/mseitzer/pytorch-fid

FID: 22.067
```
### Inception score (IS)
``` python
# https://github.com/sbarratt/inception-score-pytorch

IS: 2.045
```


### Remark
When I use GAN to generate images, we must consider to the quality of the input data. For example, if the image is rotated by 45 degrees during data augmentation, the generated photos will also be rotated by 45 degrees.

