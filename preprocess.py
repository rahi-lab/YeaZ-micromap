# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:23:25 2021

@author: gligorov
"""

import glob
import os
import os.path as op
import shutil
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

def rescale_image(
    image: Image,
    factor: float
):
    """Rescales the image by the given factor. The image is rescaled by multiplying each pixel
    by the factor. If the resulting value is greater than the maximum possible value of the image,
    the pixel is set to the maximum possible value.

    Arguments:
        image: PIL Image
        factor: factor to rescale the image by
    """
    if factor == 1.0:
        return image
    I = np.array(image)

    if I.dtype == np.uint8:  # Handle 8-bit images
        scaled_pixels = np.clip(I * factor, 0, 255)
        return Image.fromarray(scaled_pixels.astype(np.uint8))
    elif I.dtype == np.uint16:  # Handle 16-bit images
        scaled_pixels = np.clip(I * factor, 0, 65535)
        return Image.fromarray(scaled_pixels.astype(np.uint16))
    else:
        raise ValueError("Unsupported image data type")

def crop_into_patches(
    src_path: str,
    dst_path: str,
    variance_threshold: int = 500000,
    scale_factor: float = 1.0,
    patch_size: int = 256
):
    """Crops the images into patches of size patch_size x patch_size and saves them in the
    Patches folder. The patches are saved as patch_i.tif, where i is the index of the patch.
    
    Arguments:
        src_path: path to the folder containing the images
        dst_path: path to the folder where the patches will be saved
        variance_threshold: empirical varience threshold (default: 500000)
        scale_factor: factor to rescale the images by (default: 1.0)
        patch_size: size of the patches (default: 256)
    """

    # Delete the dst_path if it exists
    if op.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    half_size = patch_size//2
    crop_list = []
    image_list = [
        rescale_image(Image.open(filename), scale_factor)
        for filename in glob.glob(op.join(src_path,'*.tif'))
    ]

    for image in image_list:
        #we take the overlapping fragments
        m = int(np.floor(image.size[0]/half_size)) 
        n = int(np.floor(image.size[1]/half_size))

        for i in range(m-1):
            for j in range(n-1):
                left = i*half_size
                right = i*half_size+patch_size
                top = j*half_size
                bottom = j*half_size+patch_size
                crop_list.append(image.crop((left, top, right, bottom)))

    for i, patch in enumerate(tqdm(crop_list)):
        #empirical threshold that works very well
        if np.var(patch) > variance_threshold:
            patch.save(op.join(dst_path, f"patch_{i}.tif"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True,
                        help='Path to the folder containing the images')
    parser.add_argument('--dst_path', type=str, required=True,
                        help='Path to the folder where the patches will be saved')
    parser.add_argument('--var_thr', type=int,
                        default=500000, help='Empirical varience threshold')
    parser.add_argument('--scale_factor', type=float,
                        default=1.0, help='Factor to rescale the images by')
    parser.add_argument('--patch_size', type=int,
                        default=256, help='Size of the patches')

    args = parser.parse_args()
    crop_into_patches(
        src_path=args.src_path,
        dst_path=args.dst_path,
        variance_threshold=args.var_thr,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size
    )
