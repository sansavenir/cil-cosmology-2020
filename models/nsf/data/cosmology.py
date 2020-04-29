import torch
from torch.utils.data import Dataset
import os
import numpy as np
import imageio

class Cosmology(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.root = root

        labeled_path = os.path.join(root, 'labeled.csv')
        labeled = np.genfromtxt(labeled_path, delimiter=',', skip_header=1, dtype=np.float32)
        labeled = labeled[np.where(labeled[:, 1] > 0)]
        labeled = np.hstack([labeled, np.ones([labeled.shape[0], 1])])
        
        scored_path = os.path.join(root, 'scored.csv')
        scored = np.genfromtxt(scored_path, delimiter=',', skip_header=1, dtype=np.float32)
        scored = scored[np.where(scored[:, 1] > 0.5)]
        scored = np.hstack([scored, np.zeros([scored.shape[0], 1])])

        self.data = np.vstack([labeled, scored])

    def __getitem__(self, index):
        img_name, score, is_labeled = self.data[index]

        dir_name = 'labeled' if is_labeled else 'scored'
        img_path = os.path.join(self.root, dir_name, str(int(img_name)) + '.png')
        img = imageio.imread(img_path)
        img = np.expand_dims(img, axis=0) # HW -> CHW
        img = torch.from_numpy(img)

        # print(img_name, score, img.shape)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([score])

    def __len__(self):
        return self.data.shape[0]

