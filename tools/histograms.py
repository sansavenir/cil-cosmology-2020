import numpy as np
import imageio
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser(description='histograms')
parser.add_argument('--data_dir', type=str,
                    help='Path to the image directory')
parser.add_argument('--real', type=int, default=0,
                    help='Flag whether its real images or not')
args = parser.parse_args()

def _hist(path):
    img = imageio.imread(path)

    if img.dtype == np.float32:
        assert(np.amax(img) <= 1)
        img = (img * 255.0).astype(np.uint8)

    histogram, _ = np.histogram(img, bins=np.arange(256))
    return histogram

if args.real:
    labeled_path = os.path.join(args.data_dir, 'labeled.csv')
    labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)

    mask = labeled[:, 1] > 0
    labeled = labeled[mask]
    labeled = labeled[:500]

    paths = [os.path.join(args.data_dir, 'labeled', str(int(n)) + '.png') for n in labeled[:, 0]]
else:
    paths = glob.glob(os.path.join(args.data_dir, '*.png'))[:500]

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