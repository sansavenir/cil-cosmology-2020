from __future__ import print_function
#%matplotlib inline
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from generator import Generator
from discriminator import Discriminator
from dataset import CSVDataset
from tqdm import tqdm


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2
batch_size = 5
num_epochs = 5
lr = 0.000002
beta1 = 0.5
ngpu = 0
dataset = CSVDataset('../coords/')
nz = 10

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
netG = Generator(nz=nz, d=50).to(device)


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

netD = Discriminator(d=50).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(1, nz, device=device)

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
for epoch in tqdm(range(num_epochs)):
    # For each batch in the dataloader
    for i, data in (enumerate(dataloader, 0)):
        netD.zero_grad()
        real_cpu = data['file'].to(device).float()
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), true_label, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(true_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 500 == 0:
            with torch.no_grad():
                fixed = netG(fixed_noise).detach().cpu()
                random = netG(torch.randn(1, nz, device=device)).detach().cpu().reshape([25,2])
            # print(random)
            torch.save(netG.state_dict(), 'generator.pt')

        iters += 1

