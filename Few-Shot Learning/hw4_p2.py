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
import sys
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision import models, transforms
import re

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.Linear(128, 65)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, fix=False):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fix = fix
        if self.fix:
          for param in modelA.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2


test_csv_path = os.path.abspath(str(sys.argv[1]))
test_set_path = os.path.abspath(str(sys.argv[2]))
output_path = os.path.abspath(str(sys.argv[3]))

test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 1

label_dic = {'Alarm_Clock': 14,
 'Backpack': 27,
 'Batteries': 36,
 'Bed': 46,
 'Bike': 38,
 'Bottle': 62,
 'Bucket': 41,
 'Calculator': 26,
 'Calendar': 1,
 'Candles': 25,
 'Chair': 51,
 'Clipboards': 20,
 'Computer': 59,
 'Couch': 10,
 'Curtains': 22,
 'Desk_Lamp': 32,
 'Drill': 7,
 'Eraser': 40,
 'Exit_Sign': 11,
 'Fan': 34,
 'File_Cabinet': 31,
 'Flipflops': 24,
 'Flowers': 35,
 'Folder': 3,
 'Fork': 5,
 'Glasses': 21,
 'Hammer': 9,
 'Helmet': 23,
 'Kettle': 17,
 'Keyboard': 50,
 'Knives': 49,
 'Lamp_Shade': 30,
 'Laptop': 63,
 'Marker': 19,
 'Monitor': 42,
 'Mop': 56,
 'Mouse': 52,
 'Mug': 18,
 'Notebook': 60,
 'Oven': 47,
 'Pan': 12,
 'Paper_Clip': 58,
 'Pen': 57,
 'Pencil': 54,
 'Postit_Notes': 37,
 'Printer': 8,
 'Push_Pin': 13,
 'Radio': 6,
 'Refrigerator': 0,
 'Ruler': 43,
 'Scissors': 44,
 'Screwdriver': 64,
 'Shelf': 28,
 'Sink': 16,
 'Sneakers': 55,
 'Soda': 29,
 'Speaker': 45,
 'Spoon': 61,
 'TV': 39,
 'Table': 2,
 'Telephone': 4,
 'ToothBrush': 33,
 'Toys': 15,
 'Trash_Can': 48,
 'Webcam': 53}

class Image_Dataset(Dataset):
    def __init__(self, root, train=True, trans=None):
      self.train = train
      self.root = root
      self.transform = trans
      self.csv_file = pd.read_csv(test_csv_path)

    def __getitem__(self, index):
      self.img = Image.open(os.path.join(self.root, self.csv_file['filename'].iloc[index])).convert('RGB')
      self.img = self.transform(self.img)
      return (self.img, self.csv_file['filename'].iloc[index])
    def __len__(self):
      return len(self.csv_file)

test_set = Image_Dataset(test_set_path, train='test', trans=test_tfm)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

net = models.resnet50(pretrained=False)
ensemble = MyEnsemble(net, classifier(), fix=False)
ensemble.load_state_dict(torch.load('./resnet50_ensemble.pth?dl=1'))
model = ensemble
model = model.to(device)
model.eval()

label = []
image_id = []
id_ = []
c=0

inv_label = {v: k for k, v in label_dic.items()}

for batch in test_loader:
  imgs, labels = batch
  # Using torch.no_grad() accelerates the forward process.
  with torch.no_grad():
    logits = model(imgs.to(device)) # The shape of logits is batchsize x layer output dimension.
  # Take the class with greatest logit as prediction and record it.
  image_id.append(labels[0])
  label.append(inv_label[logits.argmax(dim=-1).cpu().numpy().tolist()[0]])
  id_.append(c)
  c+=1

outputfile = pd.DataFrame(data={'id':id_, 'filename': image_id, 'label': label})

if re.findall('csv', output_path) == []:
  output_path = os.path.join(output_path, 'pred.csv')
  outputfile.to_csv(output_path, index=False)
else:
  outputfile.to_csv(output_path, index=False)






