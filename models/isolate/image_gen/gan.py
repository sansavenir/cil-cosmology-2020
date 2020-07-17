from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from generator import Generator
from discriminator import Discriminator
from dataset import CSVDataset
from PIL import Image
from tqdm import tqdm

nz = 100

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.000002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = CSVDataset('../stars/',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5],[0.5]),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator
netG = Generator().to(device)


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

netD = Discriminator().to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(1, nz, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerG = optim.RMSprop(netG.parameters(), lr=5e-5)
optimizerD = optim.RMSprop(netD.parameters(), lr=5e-5)

fake_label = 0.
true_label = 1.


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in (range(num_epochs)):
    # For each batch in the dataloader
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        netD.zero_grad()
        real_cpu = data['image'].to(device)
        b_size = real_cpu.size(0)
        D_real = netD(real_cpu).view(-1)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        D_fake = netD(fake.detach()).view(-1)
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
        D_loss.backward()
        optimizerD.step()
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        netG.zero_grad()
        D_fake = netD(fake).view(-1)
        G_loss = -torch.mean(D_fake)
        G_loss.backward()
        D_G_z2 = D_fake.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 300 == 0:
            with torch.no_grad():
                fixed = netG(fixed_noise).detach().cpu()
                random = netG(torch.randn(1, nz, 1, 1, device=device)).detach().cpu()

            (Image.fromarray(np.uint8((fixed[0,0]+1)*127.5))).save('results/'+str(iters)+'_fixed'+'.png')
            (Image.fromarray(np.uint8((random[0,0]+1)*127.5))).save('results/'+str(iters)+'_random'+'.png')

        iters += 1
    torch.save(netG.state_dict(), 'models/' + str(epoch) + '_generator.pt')


