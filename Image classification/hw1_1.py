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

import sys
test_set_path = os.path.abspath(str(sys.argv[1]))
output_file_path = os.path.abspath(str(sys.argv[2]))

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
# I construct a dataframe to mapping image name and image class.

batch_size = 1

class Image_Dataset(Dataset):
  def __init__(self, root_dir, transforms=None):
    self.root_dir = root_dir
    self.imgname = os.listdir(self.root_dir)
    self.label_dataframe = pd.DataFrame(data={'name': self.imgname})
    self.label_dataframe['label'] = 0
    # print(self.label_dataframe)
    self.transforms = transforms

  def __len__(self):
    return len(self.label_dataframe)

  def __getitem__(self, index):
    image_path = os.path.join(self.root_dir, self.label_dataframe.iloc[index, 0])
    image = Image.open(image_path)
    
    image_label = torch.tensor(self.label_dataframe.iloc[index, 1])
    if transforms:
      image = self.transforms(image)
    
    return (image, self.label_dataframe.iloc[index, 0])

test_set = Image_Dataset(test_set_path, transforms=test_tfm)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.inception_v3(aux_logits=False, pretrained=True)
model.load_state_dict(torch.load('./inv3.pth?dl=1'))
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
outputfile.to_csv(output_file_path, index=False)
