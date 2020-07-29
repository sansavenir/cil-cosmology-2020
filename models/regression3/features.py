import numpy as np
import imageio
from tqdm import tqdm
import os
from skimage.feature import hog
from skimage import transform as tf
from skimage.feature import ORB


TILE_SIZE = 1000
NUM_TILES = (1000/250)**2
MAX_NUM_STARS = 25
FILTER_SIZE = 31
extractor = ORB(n_keypoints=10)


def get_train_features(paths, scores):
    fs = []
    ss = []
    for p, s in tqdm(list(zip(paths, scores)), desc='Generating features'):
        img = imageio.imread(p)

        for i in range(0, 1000, TILE_SIZE):
            for j in range(0, 1000, TILE_SIZE):
                t = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                f = _features_for_img(t, p)
                fs.append(f)
                ss.append(s)

    return np.asarray(fs), np.asarray(ss)


def get_pred_features(paths):
    fs = []
    for p in tqdm(paths, desc='Generating features'):
        img = imageio.imread(p)
        f = _features_for_img(img, p, scale_to_tile=True)
        fs.append(f)

    return np.asarray(fs)


def _features_for_img(img, path, scale_to_tile=False):
    px = _pixel_histogram(img)
    fft = _fft_histogram(img)
    # num_stars = _num_stars(path)
    # orb = _orb_features(img)

    fs = np.concatenate([px, fft])

    return fs


def _pixel_histogram(img):
    histogram, _ = np.histogram(img, bins=np.arange(256), density=True)

    return histogram

def _og_histogram(img):
    histogram = hog(img,
                    orientations=8,
                    pixels_per_cell=(64, 64),
                    cells_per_block=(1, 1),
                    feature_vector=True)

    return histogram


def _fft_histogram(img):
    img = img.astype(np.float32)
    eps = np.power(10.0, -15.0)
    psd_fft = np.fft.fft2(img)
    psd_fft_shift = np.abs(np.fft.fftshift(psd_fft))**2
    psd_log = 10 * np.log10(psd_fft_shift + eps)

    return np.histogram(psd_log, bins=32, range=(0, 200))[0]


def _num_stars(path):
    img_name = os.path.splitext(os.path.basename(path))[0]
    coords_path = os.path.join('../isolate/coords', img_name+'.csv')
    coords = np.load(coords_path)

    return np.asaray([coords.shape[0]])


def _orb_features(img):
  extractor.detect_and_extract(img)
  fs = np.packbits(extractor.descriptors, axis=-1)

  return np.reshape(fs, axis=0)
