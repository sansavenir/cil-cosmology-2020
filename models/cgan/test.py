import sys
sys.path.append('../')
import numpy as np
import os
import argparse
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from data.dataset import CSVDataset
from PIL import Image
from tqdm import tqdm

path = '/cluster/scratch/jkotal/cil-cosmology-2020/data/'

dataset = CSVDataset(path, scored=True, labeled=True, transform=
			transforms.Compose([
			# transforms.Resize(opt.imageSize),
			transforms.RandomCrop(878),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True)


for data in dataloader:
  img = data['image']
  img = (img.data.numpy()*127.5)+127.5
  print(img.shape)
  # (Image.fromarray(img.astype(np.uint8)[0,0])).save('a.png')
  print(img)
  print(data['label'])
  # break