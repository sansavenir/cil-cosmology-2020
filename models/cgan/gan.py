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
from disc import discriminator
from gen import generator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='beta1 for adam optimizer')
parser.add_argument('--output', default='images/', help='folder to output images and model checkpoints')

opt = parser.parse_args()


cuda = True if torch.cuda.is_available() else False

if cuda:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()


# label preprocess
img_size = 878

onehot = torch.zeros(2, 2)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
fill = torch.zeros([2, 2, img_size, img_size])
for i in range(2):
  fill[i, i, :, :] = 1


# y_gender_ = torch.LongTensor(y_gender_).squeeze()


fixed_z_ = torch.randn(1, 100).to(device)

fixed_z_ = fixed_z_.view(1, 100, 1, 1).to(device)
fixed_y_label_ = (torch.ones(1)*8.).view([1,1,1,1]).to(device)
with torch.no_grad():
  fixed_z_, fixed_y_label_ = Variable(fixed_z_), Variable(fixed_y_label_)


# training parameters
batch_size = 8
lr = opt.lr
train_epoch = 300


# data_loader

if cuda:
  path = '/cluster/scratch/jkotal/cil-cosmology-2020/data/'
else:
  batch_size = 1
  path = '../data/'

dataset = CSVDataset(path, scored=True, labeled=True, transform=
			transforms.Compose([
			# transforms.Resize(opt.size),
			# transforms.Pad(img_size-1000,0),
 			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))]))

assert dataset
train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)



G = generator(64, img_size).to(device)
D = discriminator(64).to(device)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

loss_func = torch.nn.SmoothL1Loss()
# results save folder
root = 'CelebA_cDCGAN_results/'
model = 'CelebA_cDCGAN_'
if not os.path.isdir(root):
  os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
  os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
out_path = opt.output

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
  D_losses = []
  G_losses = []
  # learning rate decay
  if (epoch + 1) == 30:
    G_optimizer.param_groups[0]['lr'] /= 5
    D_optimizer.param_groups[0]['lr'] /= 5
    print("learning rate change!")
  #
  # if (epoch + 1) == 16:
  #   G_optimizer.param_groups[0]['lr'] /= 10
  #   D_optimizer.param_groups[0]['lr'] /= 10
  #   print("learning rate change!")

  y_real_ = (torch.ones(batch_size)*8.).to(device)
  y_fake_ = torch.zeros(batch_size).to(device)
  y_real_, y_fake_ = Variable(y_real_), Variable(y_fake_)
  epoch_start_time = time.time()
  num_iter = 0
  for data in tqdm(train_loader):
    # train discriminator D
    D.zero_grad()
    G.zero_grad()

    x_ = data['image'].float().to(device)

    y_fill_ = data['label'].float().to(device)

    D_result = D(x_).view(-1)
    D_real_loss = loss_func(D_result, y_fill_.view(-1))

    z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).float().to(device)
    y_label_ = torch.from_numpy(np.random.uniform(3.,8.,(batch_size, 1,1,1))).float().to(device)
    z_, y_label_, y_fill_ = Variable(z_), Variable(y_label_), Variable(y_label_)

    G_result = G(z_, y_label_)
    # print(G_result.shape)
    D_result = D(G_result).view(-1)

    D_fake_loss = loss_func(D_result, y_fake_.view(-1))
    D_fake_score = D_result.data.mean()
    D_train_loss = D_real_loss + D_fake_loss

    D_train_loss.backward(retain_graph=True)
    D_optimizer.step()

    D_losses.append(D_train_loss.item())

    G_train_loss = loss_func(D_result, y_label_.view(-1))
    G_train_loss.backward()
    G_optimizer.step()

    G_losses.append(G_train_loss.item())
    num_iter += 1

    if (num_iter % 100) == 0:
      (Image.fromarray((G_result[0, 0].cpu().data.numpy() * 127.5 + 127.5).astype(np.uint8))).save(out_path+'random.png')
      G.eval()
      fixed = G(fixed_z_, fixed_y_label_)
      G.train()
      (Image.fromarray((fixed[0, 0].cpu().data.numpy() * 127.5 + 127.5).astype(np.uint8))).save(out_path+'fixed.png')
      print(G_train_loss.item(), D_train_loss.item())
      print('%d - %d complete!' % ((epoch + 1), num_iter))

  epoch_end_time = time.time()
  per_epoch_ptime = epoch_end_time - epoch_start_time

  (Image.fromarray((G_result[0, 0].cpu().data.numpy() * 127.5 + 127.5).astype(np.uint8))).save(out_path+str(epoch)+'_random.png')
  G.eval()
  fixed = G(fixed_z_, fixed_y_label_)
  torch.save(D.state_dict(), out_path+str(epoch)+'_model.pt')
  G.train()
  (Image.fromarray((fixed[0, 0].cpu().data.numpy() * 127.5 + 127.5).astype(np.uint8))).save(out_path+str(epoch)+'_fixed.png')

  print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
  (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
  torch.mean(torch.FloatTensor(G_losses))))
  fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
  # show_result((epoch + 1), save=True, path=fixed_p)
  train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
  train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
  train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
  pickle.dump(train_hist, f)

images = []
for e in range(train_epoch):
  img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
  images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
