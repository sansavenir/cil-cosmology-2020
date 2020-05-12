import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

class discriminator(nn.Module):
  # initializers
  def __init__(self, d=128):
    super(discriminator, self).__init__()
    self.conv1 = nn.Conv2d(1, d, 30, 10, 1)
    self.conv1_bn = nn.BatchNorm2d(d)
    self.conv2 = nn.Conv2d(d, d * 2, 16, 8, 1)
    self.conv2_bn = nn.BatchNorm2d(d * 2)
    self.conv3 = nn.Conv2d(d * 2, d * 4, 6, 2, 1)
    self.conv3_bn = nn.BatchNorm2d(d * 4)
    self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
    self.conv4_bn = nn.BatchNorm2d(d * 8)
    self.conv5 = nn.Conv2d(d * 8, d * 16, 2, 1, 0)
    self.Lin = nn.Linear(1024,1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  # def forward(self, input):
  def forward(self, input):
    x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
    x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
    x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
    x = self.conv5(x).view([-1,1024])
    x = self.Lin(x)

    return x
