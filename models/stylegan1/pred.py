# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
import glob
import pandas as pd 
import tqdm

#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

#----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.

def pred(path, query_path, output_path):
    network_pkl = misc.locate_network_pkl(path, path)
    print('Loading networks from "%s"...' % network_pkl)
    _, D, _ = misc.load_pkl(network_pkl)

    files = glob.glob(query_path)
    res = []
    for img_path in tqdm.tqdm(files):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        gimg = PIL.Image.open(img_path)
        gimg = gimg.resize((256,256), PIL.Image.ANTIALIAS)
        gimg = np.asarray(gimg)

        img = np.zeros((1,1,256,256), gimg.dtype)
        img[0,0] = gimg

        score = D.run(img, None)[0, 0]
        score = np.clip(score, 0, 8)
        res.append([img_name, score])

    df = pd.DataFrame(res, columns=['Id', 'Predicted'])
    df.to_csv(output_path, index=False)

#----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    # os.makedirs(config.result_dir, exist_ok=True)
    
    path = '/Users/Laurin/Desktop/00003-sgan-cosmology256_t-1gpu/network-snapshot-005301.pkl'
    # path = '/cluster/home/laurinb/results/stylegan1/00013-sgan-cosmology1024-1gpu/network-snapshot-004701.pkl'
    query_path = '/Users/Laurin/Documents/ETH/FS20/Computational Intelligence Lab/Project/data/query/*.png'
    output_path = '/Users/Laurin/Desktop/pred.csv'
    pred(path, query_path, output_path)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------