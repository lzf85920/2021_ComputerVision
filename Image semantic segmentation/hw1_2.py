# Import necessary packages.
import os
import pandas as pd
import numpy as np
from torch import Tensor
import torch
import random
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

import sys
test_set_path = os.path.abspath(str(sys.argv[1]))
output_file_path = os.path.abspath(str(sys.argv[2]))

classes = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown']

# RGB color for each class
colormap = [[0,255,255], [255,255,0], [255,0,255], [0,255,0], [0,0,255], [255,255,255], [0,0,0]]

len(classes), len(colormap)

# https://zhuanlan.zhihu.com/p/32506912
cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')

test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 1

class Image_Dataset(Dataset):
  def __init__(self, root_dir, trans=None):
    self.root_dir = root_dir
    self.imgs = os.listdir(self.root_dir)
    self.imgs.sort()
    self.transform = trans


  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, index):
    img = self.imgs[index]
    img = Image.open(os.path.join(self.root_dir, img))
    img = self.transform(img)  
    return img

test_set = Image_Dataset(test_set_path, trans=test_tfm)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def initialize_model(input_size, num_classes, use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained, progress=True)
    model_deeplabv3.aux_classifier = None
    model_deeplabv3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model_deeplabv3


device = "cuda" if torch.cuda.is_available() else "cpu"
n_classes = len(classes)
net= initialize_model(256, num_classes=n_classes, use_pretrained=True)
net.load_state_dict(torch.load('./Deeplab_ensemble.pth?dl=1'))
net.to(device)

pred = []
net = net.eval()
cm = np.array(colormap).astype('uint8')

for batch_idx, sample in enumerate(test_loader):
  out = net(sample.to(device))['out']
  out = F.log_softmax(out, dim=1)
  pre_label = out.max(dim=1)[1].data.cpu().numpy()
  pred.append(cm[pre_label][0])

numberlist = ["{0:04}".format(i) for i in range(500)]

size_tfm = transforms.Compose([transforms.Resize((512, 512))])

for i in range(len(pred)):
  size_tfm(Image.fromarray(pred[i])).save(os.path.join(output_file_path, '%s.png'%(numberlist[i])))
