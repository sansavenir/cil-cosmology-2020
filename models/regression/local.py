import torchvision
from torch import nn
import sys
sys.path.append('../')
import torch
from data.dataset import CSVDataset
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

cuda = True if torch.cuda.is_available() else False

if cuda:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

model = discriminator(64).to(device)

model.load_state_dict(torch.load('/cluster/home/jkotal/cil-cosmology-2020/cgan/images9/100_model.pt', map_location=device))
model.to(device)

if cuda:
  path = '/cluster/scratch/jkotal/cil-cosmology-2020/data/'
else:
  batch_size = 1
  path = '../data/'
path = '/cluster/scratch/jkotal/cil-cosmology-2020/data/'


image_size = 1000
dataset = CSVDataset(path, scored=True, transform=transforms.Compose([
                       # transforms.Resize(image_size),
                       transforms.CenterCrop(878),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5]),
                     ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                           shuffle=False, num_workers=1)
model.eval()

best = 1000
for i in tqdm(range(20,110)):
  model.load_state_dict(
  torch.load('/cluster/home/jkotal/cil-cosmology-2020/cgan/images9/'+str(i)+'_model.pt', map_location=device))
  model.to(device)
  res, lab = None, None
  for (k,data) in enumerate(loader):
    res = model(data['image'].to(device)) if res is None else torch.cat((res, model(data['image'].to(device))), 0)
    lab = data['label'] if lab is None else torch.cat((lab, data['label']), 0)
    if k > 1000:
      break
      
  if torch.norm(res-lab,2) < best:
    best = torch.norm(res-lab,2)
    print(i, torch.norm(res-lab,2))

    # print( model(data['image'].to(device)))
    # writer.writerow([data['name'][0][:-4], np.clip(model(data['image'].to(device)).item(),0.,8.)])
