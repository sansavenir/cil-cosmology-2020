import numpy as np
import os
import imageio
from skimage import transform
from tqdm import tqdm

root = '../data'
target = '../data/dataset'

os.mkdir(target)

labeled_path = os.path.join(root, 'labeled.csv')
labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)
labeled = labeled[np.where(labeled[:, 1] > 0)]
labeled = np.hstack([labeled, np.ones([labeled.shape[0], 1])])

scored_path = os.path.join(root, 'scored.csv')
scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)
scored = scored[np.where(scored[:, 1] > 0.5)]
scored = np.hstack([scored, np.zeros([scored.shape[0], 1])])

data = np.vstack([labeled, scored])

for img_name, score, is_labeled in tqdm(data):
    dir_name = 'labeled' if is_labeled else 'scored'
    img_path = os.path.join(root, dir_name, str(int(img_name)) + '.png')
    img = imageio.imread(img_path)

    new_path = os.path.join(target, str(int(img_name)) + '.png')
    new_img = transform.resize(img, (256, 256), preserve_range=True).astype(np.uint8)
    imageio.imwrite(new_path, new_img)

