import sys
sys.path.append('../../')
from tools.dataset import CSVDataset
from torchvision import datasets, transforms
import torch
import numpy as np
import operator
from PIL import Image
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--dataDir', type=str, default='/cluster/home/jkotal/cil-cosmology-2020/data/',
                    help='Flag indicating whether CUDA should be used')
parser.add_argument('--num_bg_imgs', type=int, default=1,
                    help='Number of images to be taken into account for the background random variable')
cfg = parser.parse_args()

dataset = CSVDataset(cfg.dataDir,
                     scored=False,
                     labeled=True,
                     transform=transforms.Compose([transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
filter_size = 31

os.makedirs('coords', exist_ok=True)
os.makedirs('stars', exist_ok=True)

# we generate the background from n pictures to produce a representative random variable
num_bg_imgs = cfg.num_bg_imgs
xs = np.linspace(0, 255, 256)
bs = np.zeros_like(xs)

for data in tqdm(loader):
  if data['label'] == 0.:
    continue
  img = (data['image'].data.numpy()[0,0]*255.).astype(np.uint8)
  res = np.zeros(img.shape)
  mult = np.identity(filter_size)
  nums = np.concatenate((list(range(filter_size//2)), list(range(filter_size//2,-1,-1))))
  mult *= nums
  for i in range(img.shape[0]-filter_size):
    for j in range(img.shape[0]-filter_size):
      res[i,j] = np.sum(img[i:i+filter_size,j:j+filter_size]*mult)

  res = np.where(res > 1000, res, 0)

  for i in range(0,img.shape[0],10):
    for j in range(0,img.shape[0],10):
      arg = np.unravel_index(np.argmax(res[i:i+filter_size,j:j+filter_size]), res[i:i+filter_size,j:j+filter_size].shape)
      temp = np.zeros(res[i:i+filter_size,j:j+filter_size].shape)
      temp[arg] = np.max(res[i:i+filter_size,j:j+filter_size])
      res[i:i+filter_size, j:j+filter_size] = temp

  img = img.astype(np.int32)
  for (k,a) in enumerate(np.argwhere(res>0)):
    (Image.fromarray(img[a[0]:a[0]+30,a[1]:a[1]+30].astype(np.uint8))).save('stars/'+str(data['name'].item())+'_'+str(k)+'.png')

    # fill the star patch with -1 such that it wont affect the background histogram
    img[a[0]:a[0]+30, a[1]:a[1]+30].fill(-1)

  np.savetxt('coords/'+str(data['name'].item())+'.csv', np.argwhere(res>0), delimiter=",")

  if num_bg_imgs > 0:
    # update the background histogram
    bs += np.asarray([np.count_nonzero(img == x) for x in xs])
    num_bg_imgs -= 1
  else:
    break

# average background values
bs = bs / cfg.num_bg_imgs
pk = bs / np.sum(bs)
np.save('background.npy', pk)
