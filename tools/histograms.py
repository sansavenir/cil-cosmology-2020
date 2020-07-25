import numpy as np
import imageio
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='histograms')
parser.add_argument('--data_dir', type=str,
                    help='Path to the image directory')
parser.add_argument('--scored', type=bool, default=False,
                    help='Flag whether its scored images or not')
args = parser.parse_args()

def _hist(path):
    img = imageio.imread(path)

    if img.dtype == np.float32:
        assert(np.amax(img) <= 1)
        img = (img * 255.0).astype(np.uint8)

    histogram, _ = np.histogram(img, bins=np.arange(256))
    return histogram

if args.scored:
    scored_path = os.path.join(args.data_dir, 'scored.csv')
    scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)

    mask = scored[:, 1] > 3
    scored = scored[mask]
    scored = scored[:500]

    paths = [os.path.join(args.data_dir, 'scored', str(int(n)) + '.png') for n in scored[:, 0]]
else:
    names = os.listdir(args.data_dir)
    paths = [os.path.join(args.data_dir, n) for n in names][:500]

hists = [_hist(p) for p in tqdm(paths)]

hists = np.stack(hists)
hist = np.mean(hists, axis=0)

fig = plt.figure()
plt.xlabel('brightness')
plt.ylabel('average number of pixels')
plt.yscale('log')
plt.ylim(top=10**6)
plt.ylim(bottom=0.1)

bins = list(range(255))
plt.bar(bins, hist, width=1)
plt.show()