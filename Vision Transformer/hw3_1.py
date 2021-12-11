# Import necessary packages.
import os
import pandas as pd
import numpy as np
from torch import Tensor
import torch
import random
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from pytorch_pretrained_vit import ViT
import sys
import re

test_set_path = os.path.abspath(str(sys.argv[1]))
output_path = os.path.abspath(str(sys.argv[2]))

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
# I construct a dataframe to mapping image name and image class.

batch_size = 1

class Image_Dataset(Dataset):
    def __init__(self, root, train=True, trans=None):
      self.train = train
      self.root = root
      self.transform = trans
      self.file_list = os.listdir(self.root)

    def __getitem__(self, index):
      self.img = Image.open(os.path.join(self.root, self.file_list[index])).convert('RGB')
      if (self.train == 'train') | (self.train == 'val'):
        self.img = self.transform(self.img)
        self.label = self.file_list[index].split('_')[0]
        return self.img, int(self.label)
      else:
        self.img = self.transform(self.img)
        return (self.img, self.file_list[index])
    def __len__(self):
      return len(self.file_list)

test_set = Image_Dataset(test_set_path, train='test', trans=test_tfm)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViT('B_16_imagenet1k', pretrained=True)
model.fc = nn.Linear(in_features=768, out_features=37, bias=True)
model = model.to(device)
model.load_state_dict(torch.load('./vit_v2_ensemble.pth?dl=1'))
model.to(device)

model.eval()
label = []
image_id = []
for batch in test_loader:
  imgs, labels = batch
  # Using torch.no_grad() accelerates the forward process.
  with torch.no_grad():
    logits = model(imgs.to(device)) # The shape of logits is batchsize x layer output dimension.

  # Take the class with greatest logit as prediction and record it.
  image_id.append(labels[0])
  label.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

outputfile = pd.DataFrame(data={'image_id': image_id, 'label': label})

if re.findall('csv', output_path) == []:
  output_path = os.path.join(output_path, 'pred.csv')
  outputfile.to_csv(output_path, index=False)
else:
  outputfile.to_csv(output_path, index=False)

