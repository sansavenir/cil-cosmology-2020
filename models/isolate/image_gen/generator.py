import os, time, sys
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

class Generator(nn.Module):
  # initializers
  def __init__(self, d=128):
    super(Generator, self).__init__()
    self.deconv1 = nn.ConvTranspose2d(100, d * 8, 5, 2, 1)
    self.deconv1_bn = nn.BatchNorm2d(d * 8)
    self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
    self.deconv2_bn = nn.BatchNorm2d(d * 4)
    self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 6, 2, 1)
    self.deconv3_bn = nn.BatchNorm2d(d * 2)
    # self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
    self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
    self.deconv4_bn = nn.BatchNorm2d(d)
    self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 1, 0)
    # self.Lin = nn.Linear(534*534,img_size**2)
  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  # def forward(self, input):
  def forward(self, input):
    x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)), 0.2)
    x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
    x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
    # x = F.tanh(self.deconv4(x))
    x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
    # x = self.Lin(self.deconv5(x).flatten())
    # x = torch.tanh(x.view([1,1,self.img_size,self.img_size]))
    x = torch.tanh(self.deconv5(x))
    return x