#!/usr/bin/env python3
""" Domain adaptation for segmentation of microscopy images using CycleGAN and YeaZ. 

This script performs style transfer on images in opt.dataroot,
then performs segmentation on the style transferred images.

Example:
    $ python predict.py \
        --dataroot GT_DATA_FOLDER \
        --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \
        --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \
        --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS (i.e. ./yeaz/unet/weights_budding_BF.pt) \
        --results_dir RESULTS_FOLDER \
        --threshold 0.5 \
        --epoch 200

    other options:
        --original_domain A (default) or B (i.e. if GT images are in B domain, specify B)
        --skip_style_transfer (i.e. if style transfer has already been performed, skip)
        --skip_segmentation (i.e. if segmentation has already been performed, skip)

"""

import argparse
import os
import sys
import typing

import numpy as np
import torch

sys.path.append("./cycle_gan")

from tqdm import tqdm

from cycle_gan.data import create_dataset
from cycle_gan.models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images
from yeaz.predict import YeazPredict as yeaz_predict


# TODO Merge shared code with evaluate.py

def initialzie_options() -> argparse.Namespace:
    """Initialize options
    
    Style transfer options are hard-coded for test setting. 
    Metrics options are set to default values.

    Returns:
        Initialized options
    """
    # get test options
    opt = TestOptions().parse()

    # set correct device
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    # set eval mode
    opt.eval = True

    ### Style transfer options ###
    # test code only supports num_threads = 1
    opt.num_threads = 0
    # test code only supports batch_size = 1
    opt.batch_size = 1
    # disable data shuffling; commcent this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # specify target domain
    opt.target_domain = 'B' if opt.original_domain == 'A' else 'A'
    opt.direction = 'AtoB' if opt.original_domain == 'A' else 'BtoA'

    return opt


def style_transfer(
    opt: argparse.Namespace
) -> None:
    """Perform style transfer on images in opt.dataroot

    Arguments:
        opt: Options
        epoch_range: Range of epochs to perform style transfer on
    """

    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)

    # create a model given opt.model and other options
    model = create_model(opt)

    # create a webpage for viewing the results
    print('creating web directory', opt.results_dir)
    webpage = html.HTML(opt.results_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.setup(opt)
    model.eval()

    print('Style transfer:')
    for data in tqdm(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML


def yeaz_segmentation(
    opt: argparse.Namespace,
    style_transfer_path: str
) -> None:
    """Perform segmentation on style transferred images
    
    Arguments:
        opt: Options
        epoch_range: Range of epochs to perform segmentation on
        style_transfer_path: Path to style transferred images
    """

    print('Segmentation:')

    generated_images_path = os.path.join(
        style_transfer_path, f'images/fake_{opt.target_domain}')
    image_names = [
        filename for filename in os.listdir(generated_images_path)
        if not filename.endswith('.h5')
    ]
    for image_name in tqdm(image_names):

        image_path = os.path.join(generated_images_path, image_name)
        ext = image_name.split('.')[-1]
        mask_name = image_name.replace(f'.{ext}', '_mask.h5')
        mask_path = os.path.join(generated_images_path, mask_name)
        yeaz_predict(
            image_path=image_path,
            mask_path=mask_path,
            imaging_type=None,
            fovs=[0],
            timepoints=[0, 0],
            threshold=opt.threshold,
            min_seed_dist=opt.min_seed_dist,
            weights_path=opt.path_to_yeaz_weights,
            device = opt.device
        )


def main():
    """Main function that runs: style transfer (cycle_GAN) -> segmentation (YeaZ) -> metrics (AP)"""

    # initialize style transfer options
    opt = initialzie_options()

    # run style transfer
    if not opt.skip_style_transfer:
        style_transfer(opt)

    # run yeaz segmentation
    if not opt.skip_segmentation:
        yeaz_segmentation(opt, opt.results_dir)


if __name__ == '__main__':
    main()
