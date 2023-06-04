<h1>Domain adaptation for segmentation of microscopy images using CycleGAN and YeaZ.</h1>

This repository combines the style transfer capabilities of *CycleGAN* and segmentation capabilities of *YeaZ*,
to boost the segmentation performance on out-of-domain microscopy data.

Paper: *Pan-microscope image segmentation based on a single training set*

*evaluate.py* script performs:
*  style transfer on GT images in opt.dataroot
*  segmentation on the style-transferred images, 
*  evaluates metrics on the segmented images and GT masks (from opt.dataroot).

*train_cyclegan.py* script performs:
* training of CycleGAN on GT images in opt.dataroot with different lambdas (cycle loss weights)

**Ground truth data folder structure:**
```
    GT_DATA_FOLDER
    ├── trainA
    │   ├── A1.png
    │   └── ...
    ├── trainB
    │   ├── B1.png
    │   └── ...
    ├── testA
    │   ├── A2.png
    │   └── ...
    ├── testB
    │   ├── B2.png
    │   └── ...
    ├── testA_masks
    │   ├── A2_mask.h5
    │   └── ...
    └── testB_masks
        ├── B2_mask.h5
        └── ...
```
Depending on usage, some of the folders can be empty:
* testA(_masks) and testB(_masks) can be empty during CycleGAN training
* trainA and trainB can be empty during evaluation

<h1>Usage</h1>

<h2>Evaluate pretrained CycleGAN and YeaZ</h2>
Script arguments follow the established nomenclature from two combined projects (CUT and YeaZ)

```
$ python evaluate.py \
    --dataroot GT_DATA_FOLDER \
    --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \
    --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \
    --model cycle_gan \
    --preprocess none \
    --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS (i.e. ./yeaz/unet/weights_budding_BF.pt) \
    --threshold 0.5 \
    --min_seed_dist 3 \
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
```

<h2>Train CycleGAN</h2>
Script arguments follow the established nomenclature from the CUT project (contains cycleGAN).

Adapted with minor adjustments for cycle loss lambda grid search from: https://github.com/taesungp/contrastive-unpaired-translation.


1. Start visdom server (for monitoring):
```$ python -m visdom.server```

2. Launch training:
```
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
```
See options/base_options.py and options/train_options.py for more options.</br>
If multiple lambda values are specified, a grid search will be performed.</br>
If no lambda values are specified, default values (10, 10) will be used.
