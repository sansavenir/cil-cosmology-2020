import torchvision
from torch import nn
import torch
import sys
sys.path.append('../')
from dataset import CSVDataset
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from cgan.disc import discriminator
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler



parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--device', type=str, default='cuda',
                    help='Flag indicating whether CUDA should be used')
parser.add_argument('--bs', type=int, default=16,
                    help='Batch size')
cfg = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

model = discriminator(64).to(device)
model.weight_init(mean=0.0, std=0.02)
lr = 0.00002
# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
loss_func = torch.nn.SmoothL1Loss().to(device)

image_size = 1000
batch_size = cfg.bs
workers = 4
validation_split = 0.2

if cuda:
  path = '/cluster/scratch/laurinb/cil-cosmology-2020/data/'
else:
  batch_size = 1
  path = '../data/'

# path = '/cluster/scratch/laurinb/cil-cosmology-2020/data/'

dataset = CSVDataset(path+'scored', path+'scored.csv', transform=
                       transforms.Compose([
                       # transforms.Resize(image_size),
                       # transforms.CenterCrop(image_size),
                       # transforms.Pad(1034 - 1000, 0),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5]),
                     ])
                     )

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                sampler=valid_sampler)


model.to(device)
epochs = 100
tot_loss = 0
for e in range(epochs):
  model.train()
  with tqdm(total=len(train_loader)) as pbar:
    for (step, data) in enumerate(train_loader):
      optimizer.zero_grad()

      image = data['image'].to(device)
      label = data['label'].view([batch_size, -1]).to(device).float()
      outputs = model(image)

      loss = loss_func(outputs, label)
      loss.backward()

      optimizer.step()
      tot_loss += np.sum(loss.cpu().data.numpy())
      pbar.set_description("Loss %f" % (tot_loss/((step+1))))
      pbar.update(1)

  model.eval()
  diff = 0
  for data in tqdm(validation_loader):
    image = data['image'].to(device)
    label = data['label'].view([1, -1]).to(device).float()
    outputs = model(image)
    diff += abs(label.item() - outputs.item())

  print('MAE epoch', str(e)+': ', diff / len(validation_loader))










