"""CycleGAN training script
Adapted with adjustments (for lambda grid search) from: https://github.com/taesungp/contrastive-unpaired-translation

Usage:
    Start visdom server:
        $ python -m visdom.server
    Launch training:
        $ python train_cyclegan.py \
            --model cycle_gan \
            --dataroot GT_DATA_FOLDER \
            --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \
            --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \
            --model cycle_gan \
            --preprocess crop \
            --grid_lambdas_A L1 [L2 L3 ..] (i.e. 1 10) \
            --grid_lambdas_B L1 [L2 L3 ..] (i.e. 1 10)

        other options:
            --gpu_ids GPU_ID (i.e. -1 for CPU, 0 for GPU0, 0,1 for GPU0 and GPU1)
            --batch_size BATCH_SIZE (i.e. 8)
            --n_epochs N_EPOCHS (i.e. 200)
            --n_epochs_decay N_EPOCHS_DECAY (i.e. 200)
            --lr LR (i.e. 0.0002)
            --display_freq DISPLAY_FREQ (i.e. 100)
        
        See options/base_options.py and options/train_options.py for more options.

        If multiple lambda values are specified, a grid search will be performed.
        If no lambda values are specified, default values (10, 10) will be used.
"""

import sys
import time

import torch

sys.path.append("./cycle_gan")
from cycle_gan.data import create_dataset
from cycle_gan.models import create_model

from options.train_options import TrainOptions
from util.visualizer import Visualizer

use_visualizer = True

def main(opt):
    name_prefix = opt.name
    lambdas_grid = [
        (lamA, lamB)
        for lamA in opt.grid_lambdas_A 
        for lamB in opt.grid_lambdas_B
    ]

    for lamA, lamB in lambdas_grid:
        opt.lambda_A = lamA
        opt.lambda_B = lamB
        opt.name = f"{name_prefix}_lambda_A_{lamA}_lambda_B_{lamB}"

        if use_visualizer:
            visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
            opt.visualizer = visualizer

        total_iters = 0                # the total number of training iterations
        optimize_time = 0.1
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        model = create_model(opt)      # create a model given opt.model and other options
        print('The number of training images = %d' % dataset_size)

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            if use_visualizer:
                visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

            dataset.set_epoch(epoch)
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                batch_size = data["A"].size(0)
                total_iters += batch_size
                epoch_iter += batch_size
                if len(opt.gpu_ids) > 0:
                    torch.cuda.synchronize()
                optimize_start_time = time.time()
                if epoch == opt.epoch_count and i == 0:
                    model.data_dependent_initialize(data)
                    model.setup(opt)               # regular setup: load and print networks; create schedulers
                    model.parallelize()
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                if len(opt.gpu_ids) > 0:
                    torch.cuda.synchronize()
                optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    if use_visualizer:
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    if use_visualizer:
                        visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                    if opt.display_id is None or opt.display_id > 0:
                        if use_visualizer:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)  # it's useful to occasionally show the experiment name on console
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    main(opt)