#!/usr/bin/env python3
""" Domain adaptation for segmentation of microscopy images using CycleGAN and YeaZ. 

This script performs style transfer on images in opt.dataroot,
then performs segmentation on the style transferred images, 
and finally evaluates metrics on the segmented images.

Example:
    $ python evaluate.py \
        --dataroot GT_DATA_FOLDER \
        --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \
        --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \
        --model cycle_gan \
        --preprocess none \
        --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS (i.e. ./yeaz/unet/weights_budding_BF.pt) \
        --threshold 0.5 \
        --min_seed_dist 5 \
        --min_epoch 1 \
        --max_epoch 201 \
        --epoch_step 5 \
        --results_dir RESULTS_FOLDER (i.e. D:/GAN_grid_search/results)
        --metrics_path METRICS_PATH (i.e. D:/GAN_grid_search/results/metrics.csv)

    other options:
        --original_domain A (default) or B (i.e. if GT images are in B domain, specify B)
        --skip_style_transfer (i.e. if style transfer has already been performed, skip)
        --skip_segmentation (i.e. if segmentation has already been performed, skip)
        --skip_metrics (i.e. if metrics have already been evaluated, skip)
        --metrics_patch_borders Y0 Y1 X0 X1 (i.e. 480 736 620 876)
        --plot_metrics

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
from metrics.metrics import evaluate, plot_metrics, save_metrics
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images
from yeaz.predict import YeazPredict as yeaz_predict


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

    ### Metrics options ###
    # set output metrics path if not specified
    if opt.metrics_path is None:
        opt.metrics_path = os.path.join(
            opt.results_dir, opt.name, 'metrics.csv')
    # set corect type for metrics_patch_borders
    if opt.metrics_patch_borders is not None:
        opt.metrics_patch_borders = tuple(opt.metrics_patch_borders)

    return opt


def style_transfer(
    opt: argparse.Namespace,
    epoch_range: range
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

    for epoch in epoch_range:

        opt.epoch = str(epoch)
        # create a webpage for viewing the results
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(
            opt.phase, opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
            opt.name, opt.phase, opt.epoch))

        # test with eval mode. This only affects layers like batchnorm and dropout.
        model.setup(opt)
        model.eval()

        for data in tqdm(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
        webpage.save()  # save the HTML


def yeaz_segmentation(
    opt: argparse.Namespace,
    epoch_range: range,
    style_transfer_path: str
) -> None:
    """Perform segmentation on style transferred images
    
    Arguments:
        opt: Options
        epoch_range: Range of epochs to perform segmentation on
        style_transfer_path: Path to style transferred images
    """
    for epoch in epoch_range:

        generated_images_path = os.path.join(
            style_transfer_path, f'test_{epoch}/images/fake_{opt.target_domain}')
        image_names = [
            filename for filename in os.listdir(generated_images_path)
            if not filename.endswith('.h5')
        ]

        for image_name in image_names:
            print(generated_images_path, image_name)

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


def yeaz_metrics(
    epoch_range: range,
    gt_path: str,
    style_transfer_path: str,
    original_domain: str,
    target_domain: str,
    metrics_path: str = '',
    borders: typing.Optional[tuple] = None
) -> dict:
    """Evaluate metrics on style transferred and segmented images

    Arguments:
        epoch_range: Range of epochs to evaluate metrics on
        gt_path: Path to ground truth images
        style_transfer_path: Path to style transferred images and masks
        original_domain: Domain of original images
        target_domain: Domain of style transferred images
    
    Returns:
        Dictionary of average metrics (J, SD, Jc) on segmented style transferred images for each epoch
    """
    avg_metrics_per_epoch = {}
    for epoch in epoch_range:
        J, SD, Jc = [], [], []

        generated_images_path = os.path.join(
            style_transfer_path, 'test_{}'.format(epoch), f'images/fake_{target_domain}')
        image_names = [
            filename for filename in os.listdir(generated_images_path)
            if not filename.endswith('.h5')
        ]

        for image_name in image_names:

            # get paths
            mask_name = image_name.replace('.png', '_mask.h5')
            mask_path = os.path.join(generated_images_path, mask_name)
            gt_mask_path = os.path.join(
                gt_path, f'test{original_domain}_masks', mask_name)

            # evaluate metrics
            j, sd, jc = evaluate(
                gt_mask_path,
                mask_path,
                borders=borders
            )

            J.append(j)
            SD.append(sd)
            Jc.append(jc)

        avg_metrics_per_epoch[epoch] = (
            np.mean(J), np.mean(SD), np.mean(Jc)
        )

    if metrics_path:
        save_metrics(avg_metrics_per_epoch, metrics_path)

    return avg_metrics_per_epoch


def main():
    """Main function that runs: style transfer (cycle_GAN) -> segmentation (YeaZ) -> metrics (AP)"""

    # initialize style transfer options
    opt = initialzie_options()

    # create a range of epochs to test
    epoch_range = range(opt.min_epoch, opt.max_epoch+1, opt.epoch_step)

    # run style transfer
    if not opt.skip_style_transfer:
        style_transfer(opt, epoch_range)

    style_transfer_path = os.path.join(opt.results_dir, opt.name)

    # run yeaz segmentation
    if not opt.skip_segmentation:
        yeaz_segmentation(opt, epoch_range, style_transfer_path)

    # calculate and save segmentation metrics
    if not opt.skip_metrics:
        _ = yeaz_metrics(
            epoch_range,
            opt.dataroot,
            style_transfer_path,
            opt.original_domain,
            opt.target_domain,
            opt.metrics_path,
            borders=opt.metrics_patch_borders
        )

    # plot metrics
    if opt.plot_metrics:
        loss_log_path = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')
        plot_save_path = opt.metrics_path.replace('.csv', '.png')
        plot_metrics(opt.metrics_path, loss_log_path, opt.original_domain,
                     opt.max_epoch, save_path=plot_save_path)


if __name__ == '__main__':
    main()
