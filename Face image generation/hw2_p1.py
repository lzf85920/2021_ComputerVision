# Import necessary packages.
import os
import pandas as pd
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
import random
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

import sys
output_file_path = os.path.abspath(str(sys.argv[1]))

pred = []

# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Number of channels in the training images. For color images this is 3
nc = 3

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load('./dcgan_ensemble.pth?dl=1'))

netG.eval()
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

for i in range(1000):
  noise = torch.randn(1, nz, 1, 1, device=device)
  fake = netG(noise).detach().cpu()
  pred.append(fake)


def Save_image(image_list):
  c=0
  for img in image_list:
    Image.fromarray(((np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1,2,0))).numpy()*255).astype(np.uint8)).save(os.path.join(output_file_path,'%04d.png'%c))
    c+=1

Save_image(pred)