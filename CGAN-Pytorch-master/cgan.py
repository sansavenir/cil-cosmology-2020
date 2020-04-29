import sys
sys.path.append('../')
import numpy as np
import os 
import argparse
import torch 
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torchvision import datasets 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.utils as vutils
from data.dataset import CSVDataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', required=False, help='path to data')
parser.add_argument('--bs', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=512, help='image size input')
parser.add_argument('--channels', type=int, default=1, help='number of channels')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
# parser.add_argument('--n_classes', type=int, default=10, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')
 
opt = parser.parse_args()

img_shape = (opt.channels, opt.imageSize, opt.imageSize)

cuda = True if torch.cuda.is_available() else False

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None: 
	opt.randomseed = random.randint(1,10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)



path = '/cluster/scratch/jkotal/cil-cosmology-2020/data/'
dataset = CSVDataset(path+'scored.csv', path+'scored', transform=
			transforms.Compose([
			# transforms.Resize(opt.imageSize),
			transforms.RandomCrop(opt.imageSize),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))]))

assert dataset 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.bs, shuffle=True)

# building generator
class Generator(nn.Module): 
	def __init__(self):
		super(Generator, self).__init__()
		# self.label_embed = nn.Embedding(opt.n_classes, opt.n_classes)
		self.depth=128

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.BatchNorm1d(output, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.generator = nn.Sequential(
			*init(opt.latentdim+1, self.depth),
			*init(self.depth, self.depth*2), 
			*init(self.depth*2, self.depth*4), 
			*init(self.depth*4, self.depth*8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()
			)

	# torchcat needs to combine tensors 
	def forward(self, noise, labels):
		gen_input = torch.cat((labels.view([-1,1]), noise), -1).to(device)
		img = self.generator(gen_input)
		img = img.view(img.size(0), *img_shape)
		return img


class Discriminator(nn.Module): 
	def __init__(self): 
		super(Discriminator, self).__init__()
		# self.label_embed1 = nn.Embedding(opt.n_classes, opt.n_classes)
		self.dropout = 0.4 
		self.depth = 512

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.Dropout(self.dropout))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.discriminator = nn.Sequential(
			*init(1+int(np.prod(img_shape)), self.depth, normalize=False),
			*init(self.depth, self.depth), 
			*init(self.depth, self.depth),
			nn.Linear(self.depth, 1),
			)

	def forward(self, img, labels): 
		imgs = img.view(img.size(0),-1)
		inpu = torch.cat((imgs, labels.view([-1,1])), -1).to(device)
		validity = self.discriminator(inpu)
		return validity 

# weight initialization
def init_weights(m): 
	if type(m)==nn.Linear:
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)
	

# Building generator 
generator = Generator()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Building discriminator  
discriminator = Discriminator()
discriminator.apply(init_weights)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Loss functions 
a_loss = torch.nn.SmoothL1Loss()

# Labels 
real_label = 8.
fake_label = 0.0

FT = torch.FloatTensor
FT_a = torch.FloatTensor

if cuda:
	generator.to(device)
	discriminator.to(device)
	FT = torch.cuda.FloatTensor
	FT_a = torch.cuda.FloatTensor

# training 
for epoch in range(opt.epoch): 
	for i, (data) in tqdm(enumerate(dataloader), total=len(dataset)//opt.bs):
		imgs = data['image'].float().to(device)
		labels = data['label'].float().to(device)
		batch_size = imgs.shape[0]

		# convert img, labels into proper form 
		# imgs = Variable(imgs.type(FT_a))
		# labels = Variable(labels.type(FT))
	
		# creating real and fake tensors of labels 
		reall = Variable(FT_a(batch_size,1).fill_(real_label)).to(device)
		f_label = Variable(FT_a(batch_size,1).fill_(fake_label)).to(device)

		# initializing gradient
		gen_optimizer.zero_grad() 
		d_optimizer.zero_grad()

		#### TRAINING GENERATOR ####
		# Feeding generator noise and labels 
		noise = Variable(FT_a(np.random.normal(0, 1,(batch_size, opt.latentdim)))).to(device)
		gen_labels = Variable(FT(np.random.uniform(0., 8., batch_size))).to(device)
		
		gen_imgs = generator(noise, gen_labels)
		
		# Ability for discriminator to discern the real v generated images 
		validity = discriminator(gen_imgs, gen_labels)

		# Generative loss function 
		g_loss = a_loss(validity, reall)

		# Gradients 
		g_loss.backward()
		gen_optimizer.step()

		#### TRAINING DISCRIMINTOR ####

		d_optimizer.zero_grad()

		# Loss for real images and labels
		validity_real = discriminator(imgs, labels)
		d_real_loss = a_loss(validity_real, reall)

		# Loss for fake images and labels 
		validity_fake = discriminator(gen_imgs.detach(), gen_labels)
		d_fake_loss = a_loss(validity_fake, f_label)

		# Total discriminator loss 
		d_loss = 0.5 * (d_fake_loss+d_real_loss)

		# calculates discriminator gradients
		d_loss.backward()
		d_optimizer.step()


		if i%100 == 0: 
			vutils.save_image(gen_imgs, '%s/real_samples.png' % opt.output, normalize=True)
			fake = generator(noise, gen_labels)
			vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.output, epoch), normalize=True)

	print("[Epoch: %d/%d]" "[D loss: %f]" "[G loss: %f]" % (epoch+1, opt.epoch, d_loss.item(), g_loss.item()))
	
	# checkpoints 
	torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.output, epoch))
	torch.save(discriminator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.output, epoch))















