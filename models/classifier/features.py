import numpy as np
import imageio
from tqdm import tqdm
import os
from skimage.feature import hog


TILE_SIZE = 250
NUM_TILES = (1000/250)**2
MAX_NUM_STARS = 25
FILTER_SIZE = 31


def get_train_features(paths, scores):
    fs = []
    ss = []
    for p, s in tqdm(list(zip(paths, scores)), desc='Generating features'):
        img = imageio.imread(p)

        for i in range(0, 1000, TILE_SIZE):
            for j in range(0, 1000, TILE_SIZE):
                t = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
                f = _features_for_img(t)
                fs.append(f)
                ss.append(s)

    return np.asarray(fs), np.asarray(ss)


def get_pred_features(paths):
    fs = []
    for p in tqdm(paths, desc='Generating features'):
        img = imageio.imread(p)
        f = _features_for_img(img, scale_to_tile=True)
        fs.append(f)

    return np.asarray(fs)


def _features_for_img(img, scale_to_tile=False):
    hist1 = _pixel_histogram(img)
    hist2 = _og_histogram(img)

    return np.concatenate([hist1, hist2])

    # num_stars = _num_stars(img)
    # if scale_to_tile:
    #     num_stars /= NUM_TILES
    #
    # return np.concatenate([histogram, num_stars])


def _pixel_histogram(img):
    histogram, _ = np.histogram(img, bins=np.arange(256), density=True)

    return histogram

def _og_histogram(img):
    histogram = hog(img,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    feature_vector=True)

    return histogram


def _coords(img):
    res = np.zeros(img.shape)
    mult = np.identity(FILTER_SIZE)
    nums = np.concatenate((list(range(FILTER_SIZE // 2)), list(range(FILTER_SIZE // 2, -1, -1))))
    mult *= nums
    for i in range(img.shape[0] - FILTER_SIZE):
        for j in range(img.shape[0] - FILTER_SIZE):
            res[i, j] = np.sum(img[i:i + FILTER_SIZE, j:j + FILTER_SIZE] * mult)

    res = np.where(res > 1000, res, 0)

    for i in range(0, img.shape[0], 10):
        for j in range(0, img.shape[0], 10):
            arg = np.unravel_index(np.argmax(res[i:i + FILTER_SIZE, j:j + FILTER_SIZE]), res[i:i + FILTER_SIZE, j:j + FILTER_SIZE].shape)
            temp = np.zeros(res[i:i + FILTER_SIZE, j:j + FILTER_SIZE].shape)
            temp[arg] = np.max(res[i:i + FILTER_SIZE, j:j + FILTER_SIZE])
            res[i:i + FILTER_SIZE, j:j + FILTER_SIZE] = temp

    return np.argwhere(res > 0)


def _num_stars(img):
    coords = _coords(img)

    return np.asarray([coords.shape[0] / MAX_NUM_STARS])
