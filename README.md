<h1>Domain adaptation for segmentation of microscopy images using YeaZ-micromap</h1>

<h1>Overview</h1>
YeaZ-micromap (YeaZ-microscopy mapper adapts the look of images from the target set so that they can be segmented using a neural network trained for the segmentation of images from the source set.
The tool consists of two modules: the style transfer module, which maps the look of images; and the YeaZ module, which segments the style-transferred images. As the style-transfer module, we used a generative adversarial network called CycleGAN. 

<p>

Paper: *Pan-microscope image segmentation based on a single training set*
<p align="center">
    
<img src="https://github.com/rahi-lab/YeaZ-micromap/assets/48595116/8ad9fb06-d23e-4afe-a34a-638251835131"/>

</p>



<h1>Installation</h1>
<h4>System requirements</h4>
The code was written in Python 3.9 and tested on Windows 11 and RedHat Linux server 7.7.
<h4>Installation steps</h4>

Installation time is less than 10 minutes.
</br>

 1. Clone the repository (```git clone https://github.com/rahi-lab/YeaZ-micromap```) or download it directly from the GitHub webpage
 2. Create a virtual environment ```conda create -n YeaZ-micromap python=3.9```
 3. Activate the virtual environment ```conda activate YeaZ-micromap```
 4. Install required packages ```pip install -r requirements.txt```


<h1>Usage</h1>

The code can be run from the command line and is split into two parts: 1. Training of the microscopy style-transfer using CycleGAN 2. Evaluation of the training by segmenting the mapped images using a pre-trained YeaZ network for segmentation. More specifically:

1. *train_cyclegan.py* script performs:
* style transfer training between the images in the trainA and trainB folders

2. *evaluate.py* script performs:
*  style transfer on source dataset images in one of the specified folders (testA or testB) using the pretrained CycleGAN
*  segmentation on the style-transferred images using the pretrained YeaZ weights, 
*  evaluation of segmentation quality based on the segmented images and GT masks

Both parts of the code rely on the following folder structure:
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
Depending on the usage, some of the folders can be empty:
* testA(_masks) and testB(_masks) can be empty during CycleGAN training
* trainA and trainB can be empty during the evaluation step


<h3>Train CycleGAN</h3>
Script arguments follow the established options nomenclature from the original cycleGAN repository (https://github.com/taesungp/contrastive-unpaired-translation). For more details see the comments in the code below.


1. Start visdom server:
```$ python -m visdom.server```
<p> Visdom is a visualization tool that communicates with the CycleGAN code during training and saves one example of mapping per epoch. This is useful for quickly checking whether the mapping qualitatively  makes sense. Saved data can be later accessed using an HTML interface, in Checkpoint/Experiment_Name/web/index.html</p>


2. Launch training:
```
    $ python train_cyclegan.py \
        --model cycle_gan \ #The generative model we use for transfer of the look of microscopy images
        --dataroot GT_DATA_FOLDER \ #Directory that contains images used for training
        --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER (i.e. D:/GAN_grid_search) \ #Directory where the trained models are saved. By default,  models will be saved after every epoch
        --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING (i.e. cyclegan_lambda_A_100_lambda_B_10_trial_2) \ #Name of the experiment (to be used during the prediction phase)
        --preprocess crop \ #Preprocess images by cropping them to small patches, default = 256 px X 256 px
        --grid_lambdas_A L1 L2 L3 .. (i.e. 1 10) \ #cycle_consistency_loss weights used for A->B->A mapping, default = 10
        --grid_lambdas_B L1 L2 L3 .. (i.e. 1 10) #cycle_consistency_loss weights used for B->A->B mapping, default = 10

    other options:
        --gpu_ids GPU_ID # -1 for CPU; 0 for GPU0; 0,1 for GPU0 and GPU1, default = 0
        --batch_size BATCH_SIZE #default = 1
        --n_epochs N_EPOCHS #default = 200
        --n_epochs_decay N_EPOCHS_DECAY #Number of epochs before learning rate linearly decays to 0, default = 200
        --lr LR #Initial learning rate for adam, default = 0.0002

```
If multiple lambda values are specified, a grid search will be performed.</br>
If no lambda values are specified, default values (10, 10) will be used.

<h3>Evaluate the mapping using pretrained YeaZ</h3>
<p> For evaluating the segmentation accuracy, the user provides the directory with checkpoint weights from the CycleGAN training ("checkpoints_dir"), the DNN weights used for training of the source dataset ("path_to_yeaz_weights"), among other things. The rest of the arguments refer to either other trained CycleGAN specifications ("dataroot", "name", "model", "preprocess") or to YeaZ segmentation ("threshold", "min_seed_dist", "min_epoch", "max_epoch", "epoch_step"). The dataroot folder contains the mask of the small annotated patch of the test image for only one of the domains (corresponding to the target set). If specified, a subpart (patch) of the big mask can be used for training evaluation instead of the whole mask. In that case "metrics_patch_borders" should be supplied as an additional parameter. The resulting segmentation masks will be saved in "results_dir" and the metrics of segmentation in "metrics_path". </p>

```
$ python evaluate.py \
    --dataroot GT_DATA_FOLDER \ #Directory that contains test images
    --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER \ #Checkpoints directory as specified during the CycleGAN training, e.g. D:/GAN_grid_search
    --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING  \ #Experiment name as specified during the CycleGAN training, e.g. cyclegan_lambda_A_100_lambda_B_10_trial_1
    --model cycle_gan \ 
    --preprocess none \
    --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS \ #Pretrained YeaZ weights, e.g. ./yeaz/unet/weights_budding_BF.pt
    --threshold 0.5 \ #Threshold used during YeaZ prediction, default = 0.5
    --min_seed_dist 3 \ #Minimal seed distance between two cells used during YeaZ prediction, default = 5
    --min_epoch 1 \ #First CycleGAN epoch to take into consideration for evaluation
    --max_epoch 201 \ #Last CycleGAN epoch to take into consideration for evaluation
    --epoch_step 5 \ #Evaluate every n-th epoch of the CycleGAN training
    --results_dir RESULTS_FOLDER (i.e. ) #Output folder where style-transferred and segmented images will be saved, e.g. D:/GAN_grid_search/results
    --metrics_path METRICS_PATH (i.e. ) #Output folder where metrics (AP) will be saved, e.g. D:/GAN_grid_search/results/metrics.csv

other options:
    --original_domain A or B #Source dataset to use test sets from, default = A
    --skip_style_transfer #i.e. if style transfer has already been performed, skip
    --skip_segmentation #i.e. if segmentation has already been performed, skip
    --skip_metrics #i.e. if metrics have already been evaluated, skip
    --metrics_patch_borders Y0 Y1 X0 X1 #e.g. 480 736 620 876
    --plot_metrics
```

<h1>Demo</h1>
