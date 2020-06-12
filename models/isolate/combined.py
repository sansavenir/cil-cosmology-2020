import sys
sys.path.append('../')
from coord_gen.generator import Generator as cg
from image_gen.generator import Generator as ig
import torch
import numpy as np
from PIL import Image
import glob

cg = cg(nz=10,d=50)
cg.load_state_dict(torch.load('coord_gen/generator.pt'))
cg.eval()

ig = ig()
ig.load_state_dict(torch.load('image_gen/models/45_generator.pt'))
ig.eval()

backgrounds = glob.glob('background/*.png')

for i in range(10):
  bcks = np.random.choice(backgrounds, 100)
  res = np.zeros((1000,1000), np.uint8)
  for (k,b) in enumerate(bcks):
    if k%2==0:
      res = np.maximum(res, np.asarray(Image.open(b)))
    else:
      res = np.minimum(res, np.asarray(Image.open(b)))

  coords = (cg(torch.randn(1, 10)).detach().cpu().data.numpy().reshape([25, 2])*1000).astype(int)

  for c in coords:
    if c[0]< 0 or c[1] < 0:
      continue
    star = ig(torch.randn(1, 100, 1, 1)).detach().cpu().data.numpy()[0,0,:30,:30]
    res[c[0]:c[0]+30, c[1]:c[1]+30] = np.maximum(star*127.5+127.5, res[c[0]:c[0]+30, c[1]:c[1]+30])

  (Image.fromarray(res)).save('results/'+str(i)+'.png')
