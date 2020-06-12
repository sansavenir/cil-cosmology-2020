import torch.nn as nn
import torch.nn.functional as F
import torch

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

class Discriminator(nn.Module):
  # initializers
  def __init__(self, d=128, num=25):
    super(Discriminator, self).__init__()
    self.Lin1 = nn.Linear(num*2, d)
    self.Lin2 = nn.Linear(d, 2*d)
    self.Lin3 = nn.Linear(2*d, 1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  # def forward(self, input):
  def forward(self, input):
    x = self.Lin1(input)
    x = self.Lin2(x)
    x = torch.sigmoid(self.Lin3(x))
    return x
