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
  def __init__(self, nz=100, d=128, out=25):
    super(Generator, self).__init__()
    self.Lin1 = nn.Linear(nz, d)
    self.Lin2 = nn.Linear(d, 2*d)
    self.Lin3 = nn.Linear(2*d, out*2)
  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  # def forward(self, input):
  def forward(self, input):
    x = self.Lin1(input)
    x = self.Lin2(x)
    x = self.Lin3(x)
    x = torch.tanh(x)
    return x