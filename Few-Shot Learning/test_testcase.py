import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosin_similarity(query_a, proto_b):
  cos_s = []
  # Dot and norm
  for i in query_a:

    dot = torch.matmul(i.reshape(1, -1), torch.transpose(proto_b, 0, 1))
    norm_q = torch.norm(i)
    norm_p = torch.norm(proto_b)
    cos_sim = dot / (norm_q*norm_p)
    cos_s.append(cos_sim)

  cos_sim = torch.tensor([item[0].cpu().detach().numpy() for item in cos_s]).cuda() 

  return cos_sim

def predict(args, model, data_loader):
    pred_result = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:].cuda() 
            query_input   = data[args.N_way * args.N_shot:,:,:,:].cuda()

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            x = model(support_input)
            # print('model output support', x.shape)
            x = x.reshape(args.N_shot, args.N_way, -1).mean(dim=0)
            p = x
            # print(p.shape)
            if args.dis_metrix == 'euclidean':
              logits = euclidean_metric(model(query_input), p)
            elif args.dis_metrix == 'cosin_similarity':
              logits = cosin_similarity(model(query_input), p)
            elif args.dis_metrix == 'parametric':
              para_input = torch.matmul(model(query_input), torch.transpose(p, 0, 1))
              para_input = para_input.reshape(1, -1)
              logits = para_model(para_input)
              logits = logits.reshape(args.N_query*args.N_way, args.N_way)
            else:
              pass
            pred = torch.argmax(logits, dim=1)
            pred = pred.tolist()
            sys.breakpointhook
            pred_result.append(pred)
        pred_result = np.array(pred_result)

    return pred_result

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Parametric(nn.Module):

    def __init__(self, query, train_way):
        super().__init__()
        self.para_dis = nn.Sequential(
          nn.Linear(query*train_way*train_way, 64),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, query*train_way*train_way)
          )

    def forward(self, x):
      x = self.para_dis(x)
      return x


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--dis_metrix', type=str, default='euclidean')
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    prediction_results = predict(args, model, test_loader)
    # TODO: output your prediction to csv

    column_name = ['query%s'%i for i in range(75)]

    pd.DataFrame(data=prediction_results, columns=column_name).to_csv(args.output_csv)