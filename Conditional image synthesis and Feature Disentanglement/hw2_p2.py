# Import necessary packages.
import os
import pandas as pd
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
from torch.autograd import Variable
import random
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

import sys
output_file_path = os.path.abspath(str(sys.argv[1]))

# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Number of channels in the training images. For color images this is 3
nc = 3
num_classes = 10
image_size = 32

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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

generator = Generator(nz, num_classes, image_size).cuda()
generator.load_state_dict(torch.load('./acgan_ensemble.pth?dl=1'))

generator.eval()
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def pred_image(number_sample, classes, noise_z):
  labels = np.array([(i*0) + classes for i in range(number_sample)])
  labels = Variable(LongTensor(labels))
  gen_imgs = generator(noise_z, labels)
  return gen_imgs

image_name = ["%03d" % x for _ in range(10) for x in range(1, 101)]

np.random.seed(0)
z = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, nz))))

c = 0
for j in range(10):
  pred_data = pred_image(number_sample=100, classes=j, noise_z=z)
  for i in pred_data:
    predimg = Image.fromarray(((np.transpose(vutils.make_grid(i.cpu(),nrow=10, padding=2, normalize=True), (1,2,0))).numpy()*255).astype(np.uint8))
    predimg = transforms.Resize((28,28))(predimg)
    predimg.save(os.path.join(output_file_path,'./%s_%s.png'%(j, image_name[c])))
    c += 1








