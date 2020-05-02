import os, time, sys
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from data.dataset import CSVDataset
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

class generator(nn.Module):
  # initializers
  def __init__(self, d=128, img_size=100):
    super(generator, self).__init__()
    self.deconv1_1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
    self.deconv1_1_bn = nn.BatchNorm2d(d * 4)
    self.deconv1_2 = nn.ConvTranspose2d(1, d * 4, 4, 1, 0)
    self.deconv1_2_bn = nn.BatchNorm2d(d * 4)
    self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
    self.deconv2_bn = nn.BatchNorm2d(d * 4)
    self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
    self.deconv3_bn = nn.BatchNorm2d(d * 2)
    # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
    self.deconv4 = nn.ConvTranspose2d(d * 2, d, 8, 4, 1)
    self.deconv4_bn = nn.BatchNorm2d(d)
    self.deconv5 = nn.ConvTranspose2d(d, 1, 16, 8, 1)
    # self.Lin = nn.Linear(534*534,img_size**2)
    self.img_size = img_size

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  # def forward(self, input):
  def forward(self, input, label):
    x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
    y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
    x = torch.cat([x, y], 1)
    x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
    x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
    # x = F.tanh(self.deconv4(x))
    x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
    # x = self.Lin(self.deconv5(x).flatten())
    # x = torch.tanh(x.view([1,1,self.img_size,self.img_size]))
    x = torch.tanh(self.deconv5(x))
    return x


def show_result(num_epoch, show=False, save=False, path='result.png'):
  G.eval()
  test_images = G(fixed_z_.to(device), fixed_y_label_.to(device))[:-1].view([-1,img_size,img_size])
  G.train()

  size_figure_grid = 4
  fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
  for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)

  for k in range(size_figure_grid * size_figure_grid - 1):
    i = k // size_figure_grid
    j = k % size_figure_grid
    ax[i, j].cla()
    ax[i, j].imshow((test_images[k].cpu().data.numpy()+ 1) / 2)

  label = 'Epoch {0}'.format(num_epoch)
  fig.text(0.5, 0.04, label, ha='center')

  if save:
    plt.savefig(path)

  if show:
    plt.show()
  else:
    plt.close()

def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
  x = range(len(hist['D_losses']))

  y1 = hist['D_losses']
  y2 = hist['G_losses']

  plt.plot(x, y1, label='D_loss')
  plt.plot(x, y2, label='G_loss')

  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.legend(loc=4)
  plt.grid(True)
  plt.tight_layout()

  if save:
    plt.savefig(path)

  if show:
    plt.show()
  else:
    plt.close()

def show_noise_morp(show=False, save=False, path='result.png'):
  source_z_ = torch.randn(10, 100)
  z_ = torch.zeros(100, 100)
  for i in range(5):
    for j in range(10):
      z_[i * 20 + j] = (source_z_[i * 2 + 1] - source_z_[i * 2]) / 9 * (j + 1) + source_z_[i * 2]

  for i in range(5):
    z_[i * 20 + 10:i * 20 + 20] = z_[i * 20:i * 20 + 10]

  y_ = torch.cat([torch.zeros(10, 1), torch.ones(10, 1)], 0).type(torch.LongTensor).squeeze()
  y_ = torch.cat([y_, y_, y_, y_, y_], 0)
  y_label_ = onehot[y_]
  z_ = z_.view(-1, 100, 1, 1)
  y_label_ = y_label_.view(-1, 2, 1, 1)

  with torch.no_grad():
    z_, y_label_ = Variable(z_), Variable(y_label_)

  G.eval()
  test_images = G(z_, y_label_)
  G.train()

  size_figure_grid = 10
  fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(img_size, img_size))
  for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)

  for k in range(10 * 10):
    i = k // 10
    j = k % 10
    ax[i, j].cla()
    ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

  if save:
    plt.savefig(path)

  if show:
    plt.show()
  else:
    plt.close()