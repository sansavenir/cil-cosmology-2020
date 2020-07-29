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
import os

nz = 100

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

batch_size = 64
num_epochs = 50

dataset = CSVDataset('../stars/',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5],[0.5]),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator
netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(1, nz, 1, 1, device=device)

optimizerG = optim.RMSprop(netG.parameters(), lr=5e-5)
optimizerD = optim.RMSprop(netD.parameters(), lr=5e-5)

fake_label = 0.
true_label = 1.

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


