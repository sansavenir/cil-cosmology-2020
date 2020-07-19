import torchvision
from torch import nn
import sys
sys.path.append('../')
import torch
from dataset import CSVDataset
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import csv
from cgan.disc import discriminator
import numpy as np


parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--device', type=str, default='cuda',
                    help='Flag indicating whether CUDA should be used')
parser.add_argument('--bs', type=int, default=16,
                    help='Batch size')
cfg = parser.parse_args()

device = torch.device(cfg.device)

# model = torchvision.models.resnet18(pretrained=False)
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(512, 1)

cuda = True if torch.cuda.is_available() else False

if cuda:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

model = discriminator(64).to(device)

model.load_state_dict(torch.load('/cluster/home/laurinb/cil-cosmology-2020/cgan/images9/100_model.pt', map_location=device))
model.to(device)

if cuda:
  path = '/cluster/scratch/laurinb/cil-cosmology-2020/data/'
else:
  batch_size = 1
  path = '../data/'
path = '/cluster/scratch/laurinb/cil-cosmology-2020/data/'


image_size = 1000
dataset = CSVDataset(path+'query', transform=transforms.Compose([
                       # transforms.Resize(image_size),
                       # transforms.CenterCrop(878),
                       # transforms.RandomCrop(878),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5]),
                     ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=False, num_workers=1)
with open('result.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["Id", 'Predicted'])
  for data in tqdm(loader):
    # print( model(data['image'].to(device)))
    writer.writerow([data['name'][0][:-4], np.clip(model(data['image'].to(device)).item(),0.,8.)])
