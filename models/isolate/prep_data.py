import sys
sys.path.append('../../')

from tools.dataset import CSVDataset
from torchvision import datasets, transforms
import torch
import numpy as np
import operator
from PIL import Image
from tqdm import tqdm

dataset = CSVDataset('../../data/', scored=False, labeled=True, transform=
			transforms.Compose([
 			transforms.ToTensor(),
      ]
      ))

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
filter_size = 31
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

  # print(np.count_nonzero(res))
  for (k,a) in enumerate(np.argwhere(res>0)):
    (Image.fromarray(img[a[0]:a[0]+30,a[1]:a[1]+30])).save('stars/'+str(data['name'].item())+'_'+str(k)+'.png')
    img[a[0]:a[0]+30, a[1]:a[1]+30].fill(0)

  np.savetxt('coords/'+str(data['name'].item())+'.csv', np.argwhere(res>0), delimiter=",")
  (Image.fromarray(img.astype(np.uint8))).save('background/'+str(data['name'].item())+'.png')
