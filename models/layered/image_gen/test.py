from generator import Generator
import torch
from PIL import Image
import numpy as np

generator = Generator()
generator.load_state_dict(torch.load('30_generator.pt'))
generator.eval()

for i in range(100):
  star = generator(torch.randn(1, 100, 1, 1)).detach().cpu().data.numpy()[0, 0, :30, :30]
  star = (star*127.5+127.5).astype(np.uint8)
  (Image.fromarray(star)).save('tests/'+str(i)+'.png')

