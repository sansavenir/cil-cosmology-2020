import sys
sys.path.append('../')
import os
from coord_gen.generator import Generator as cg
from image_gen.generator import Generator as ig
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='reg')
parser.add_argument('--coordModel', type=str, default='coord_gen/generator.pt',
                    help='Flag indicating whether CUDA should be used')
parser.add_argument('--starModel', type=str, default='image_gen/models/46_generator.pt',
                    help='Flag indicating whether CUDA should be used')
cfg = parser.parse_args()

os.makedirs('results', exist_ok=True)

cg = cg(nz=10,d=50)
cg.load_state_dict(torch.load(cfg.coordModel))
cg.eval()

ig = ig()
ig.load_state_dict(torch.load(cfg.starModel))
ig.eval()

pk = np.load('background.npy')
xs = np.linspace(0, 255, 256)
bg_rv = stats.rv_discrete(name='bg', values=(xs, pk))

for i in tqdm(range(500)):
  res = np.random.uniform(size=(1000, 1000))
  res = bg_rv.ppf(res).astype(np.uint8)

  coords = (cg(torch.randn(1, 10)).detach().cpu().data.numpy().reshape([25, 2])*1000).astype(int)

  for c in coords:
    if c[0]< 0 or c[1] < 0:
      continue
    star = ig(torch.randn(1, 100, 1, 1)).detach().cpu().data.numpy()[0,0,:30,:30]
    res[c[0]:c[0]+30, c[1]:c[1]+30] = np.maximum(star*127.5+127.5, res[c[0]:c[0]+30, c[1]:c[1]+30])

  (Image.fromarray(res)).save('results/'+str(i)+'.png')
