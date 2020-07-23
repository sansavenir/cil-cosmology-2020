import numpy as np
import imageio
from tqdm import tqdm


def get_features(paths):
    fs = [_features_for_path(p) for p in tqdm(paths, desc='Generating features')]
    return np.stack(fs)


def _features_for_path(path):
    img = imageio.imread(path)
    histogram = _pixel_histogram(img)

    return histogram


def _pixel_histogram(img):
    histogram, _ = np.histogram(img, bins=np.arange(256), density=True)

    return histogram