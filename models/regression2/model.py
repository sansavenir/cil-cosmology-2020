import torch.nn as nn
import torch.nn.functional as F
import torch

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Scorer(nn.Module):

    def __init__(self, d_img=32, d_pix=255):
        super(Scorer, self).__init__()

        self.conv1 = nn.Conv2d(1, d_img, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d_img)
        self.conv2 = nn.Conv2d(d_img, d_img * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d_img * 2)
        self.conv3 = nn.Conv2d(d_img * 2, d_img * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d_img * 4)
        self.conv4 = nn.Conv2d(d_img * 4, d_img * 8, 3, 1, 0)

        self.fc1 = nn.Linear(d_pix, 32)
        self.fc2 = nn.Linear(230432, 32)
        self.fc3 = nn.Linear(32, 1)


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, img, pix):
        x = F.leaky_relu(self.fc1(pix), 0.2)
        x = self.fc3(x)

        return x


        bs = img.shape[0]

        x1 = F.leaky_relu(self.conv1_bn(self.conv1(img)), 0.2)
        x1 = F.leaky_relu(self.conv2_bn(self.conv2(x1)), 0.2)
        x1 = F.leaky_relu(self.conv3_bn(self.conv3(x1)), 0.2)
        x1 = self.conv4(x1).view([bs, -1])

        x2 = F.leaky_relu(self.fc1(pix), 0.2)

        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)

        return x
