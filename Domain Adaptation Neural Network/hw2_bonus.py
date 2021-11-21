import os
import pandas as pd
import numpy as np
import re
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
from torch.autograd import Variable
from torch.autograd import Function
import random
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

import sys

test_set_path = os.path.abspath(str(sys.argv[1]))
t_name = str(sys.argv[2])
output_file_path = os.path.abspath(str(sys.argv[3]))

class Image_Dataset(Dataset):
  def __init__(self, root_dir, transforms=None):
    self.root_dir = root_dir
    self.imgname = os.listdir(self.root_dir)
    self.imgname.sort()
    self.label_dataframe = pd.DataFrame(data={'name': self.imgname})
    self.label_dataframe['label'] = 0
    self.transforms = transforms

  def __len__(self):
    return len(self.label_dataframe)

  def __getitem__(self, index):
    image_path = os.path.join(self.root_dir, self.label_dataframe.iloc[index, 0])
    image = Image.open(image_path)
    
    image_label = torch.tensor(self.label_dataframe.iloc[index, 1])
    if transforms:
      image = self.transforms(image)
    
    return image, self.label_dataframe.iloc[index, 0]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SVHNmodel(nn.Module):
    """ SVHN architecture
    """

    def __init__(self, batchnorm, dropout):
        super(SVHNmodel, self).__init__()
        self.restored = False
        self.batchnorm = batchnorm
        self.dropout = dropout

        if self.batchnorm & self.dropout:
          self.feature = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),  # 28
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 13
              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),  # 9
              nn.BatchNorm2d(64),
              nn.Dropout2d(),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 4
              nn.ReLU(inplace=True),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),  # 1
          )
        else:
          self.feature = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),  # 28
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 13
              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),  # 9
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 4
              nn.ReLU(inplace=True),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),  # 1
          )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, d_classifier, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        if d_classifier:
          domain_output = self.discriminator(reverse_feature)

          return class_output, domain_output
        else:
          return class_output

class MNISTMmodel(nn.Module):
    """ MNIST-M architecture
    """
    def __init__(self):
        super(MNISTMmodel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.Linear(100, 2),
        )

    def forward(self, d_classifier, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        if d_classifier:
          domain_output = self.discriminator(reverse_feature)

          return class_output, domain_output
        else:
          return class_output

class USPSMmodel(nn.Module):
    """ USPS architecture"""
    def __init__(self):
        super(USPSMmodel, self).__init__()
        self.restored = False
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 1 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=24,
                      kernel_size=(5, 5)),  # 24 8 8
            nn.BatchNorm2d(24),
            nn.Dropout2d(0.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 24 4 4
            )

        self.classifier = nn.Sequential(
            nn.Linear(24*4*4, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 10),
            )

        self.discriminator = nn.Sequential(
            nn.Linear(24*4*4, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 2),
            )

    def forward(self, d_classifier, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], input_data.data.shape[1], 28, 28)     
        feature = self.feature(input_data)
        feature = feature.view(-1, feature.shape[1] * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        if d_classifier:
          domain_output = self.discriminator(reverse_feature)

          return class_output, domain_output
        else:
          return class_output

device = "cuda" if torch.cuda.is_available() else "cpu"

if (t_name == 'MNIST-M') or (t_name == 'mnistm'):
  dann = SVHNmodel(batchnorm=True, dropout=True)
  dann.load_state_dict(torch.load('./DANN_SVHN_to_MNIST-M_improved_best.pth?dl=1'))
  print('load model from : DANN_SVHN_to_MNIST-M_improved_best.pth?dl=1' )
  dann.to(device)
  test_tfm = transforms.Compose([
  transforms.Resize((32,32)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])
elif (t_name == 'USPS') or (t_name == 'usps'):
  dann = MNISTMmodel()
  dann.load_state_dict(torch.load('./MNIST-M_to_USPS_improved_best.pth?dl=1'))
  print('load model from : MNIST-M_to_USPS_improved_best.pth?dl=1' )
  dann.to(device)
  test_tfm = transforms.Compose([
  transforms.Resize((28,28)),
  transforms.Grayscale(3),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])
else:
  dann = USPSMmodel()
  dann.feature[4] = nn.Conv2d(32, 24, kernel_size=(5, 5), stride=(1, 1))
  dann.feature[5] = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  dann.feature[6] = nn.Dropout2d(p=0.5, inplace=False)

  dann.classifier[0] = nn.Linear(in_features=24*4*4, out_features=24, bias=True)
  dann.classifier[1] = nn.BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  dann.classifier[2] = nn.Sigmoid()
  dann.classifier[3] = nn.Linear(in_features=24, out_features=10, bias=True)

  dann.discriminator[0] = nn.Linear(in_features=24*4*4, out_features=24, bias=True)
  dann.discriminator[1] = nn.BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  dann.discriminator[2] = nn.Sigmoid()
  dann.discriminator[3] = nn.Linear(in_features=24, out_features=2, bias=True)
  dann.load_state_dict(torch.load('./USPS_to_SVHN_improved_best.pth?dl=1'))
  dann.to(device)
  print('load model from : USPS_to_SVHN_improved_best.pth?dl=1' )
  test_tfm = transforms.Compose([
    transforms.CenterCrop((28, 20)),
    transforms.Resize((28,28)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


test_set = Image_Dataset(test_set_path, transforms=test_tfm)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# labels = []
# image_id = []

# for batch in test_loader:
#   img, label = batch
#   logits = dann(True, img.to(device))[0]
#   labels.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
#   image_id.append(label)

def test(model, data_loader, device, output_path):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()
    lab = []
    i_id = []
    for (images, imagename) in data_loader:
      
        for i in imagename:
          i_id.append(i)
        images = images.to(device)
        preds, domain = model(d_classifier=True, input_data=images, alpha=0)
        pred_cls = preds.data.max(1)[1]
        for i in pred_cls:
          lab.append(i.cpu().item())
    
    outputfile = pd.DataFrame(data={'image_name': i_id, 'label': lab})
    if re.findall('csv', output_path) == []:
      output_path = os.path.join(output_path, 'test_pred.csv')
      outputfile.to_csv(output_path, index=False)
    else:
      outputfile.to_csv(output_path, index=False)

test(dann.to(device), test_loader, device, output_file_path)
# image_id = [name for batch_name in image_id for name in batch_name]
# outputfile = pd.DataFrame(data={'image_name': image_id, 'label': labels})

# if re.findall('csv', output_file_path) == []:
#   output_file_path = os.path.join(output_file_path, 'test_pred.csv')
#   outputfile.to_csv(output_file_path, index=False)
# else:
#   outputfile.to_csv(output_file_path, index=False)


































