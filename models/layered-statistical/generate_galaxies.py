import sys
sys.path.append('../')
import os
from coord_gen.generator import Generator as cg
from image_gen.generator import Generator as ig
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--coordModel', type=str, default='coord_gen/generator.pt',
                    help='Flag indicating whether CUDA should be used')
parser.add_argument('--starModel', type=str, default='image_gen/models/300_generator.pt',
                    help='Flag indicating whether CUDA should be used')
cfg = parser.parse_args()

os.makedirs('results', exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load layer checkpoints for coordinates and stars
cg = cg(nz=10,d=50).to(device)
cg.load_state_dict(torch.load(cfg.coordModel))
cg.eval()

ig = ig().to(device)
ig.load_state_dict(torch.load(cfg.starModel))
ig.eval()

# load random variable for background noise
pk = np.load('background.npy')
xs = np.linspace(0, 255, 256)
bg_rv = stats.rv_discrete(name='bg', values=(xs, pk))

for i in tqdm(range(5000)):
  # first we generate a noisy background according to the random background variable
  res = np.random.uniform(size=(1000, 1000))
  res = bg_rv.ppf(res).astype(np.uint8)

  # generate the coordinates of the stars
  coords = (cg(torch.randn(1, 10).to(device)).detach().cpu().data.numpy().reshape([25, 2])*970).astype(int)

  for c in coords:
    if c[0]< 0 or c[1] < 0:
      continue

    # generate a new star from a random vector
    star = ig(torch.randn(1, 100, 1, 1).to(device)).detach().cpu().data.numpy()[0,0,:30,:30]

    # lay the star image on top of the random variable
    res[c[0]:c[0]+30, c[1]:c[1]+30] = np.maximum(star*127.5+127.5, res[c[0]:c[0]+30, c[1]:c[1]+30])

  (Image.fromarray(res)).save('results/'+str(i)+'.png')
