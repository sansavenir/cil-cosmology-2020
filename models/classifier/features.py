import numpy as np
import imageio
from tqdm import tqdm

def get_features(paths):
    fs = []

    for path in tqdm(paths, desc='Generating features'):
        img = imageio.imread(path)
        histogram = _pixel_histogram(img)
        fs.append(histogram)

    return np.stack(fs)


def _pixel_histogram(img):
    histogram, _ = np.histogram(img, bins=np.arange(256), density=True)

    return histogram