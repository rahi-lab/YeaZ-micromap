<h1>Domain adaptation for segmentation of microscopy images using YeaZ-micromap</h1>

<h1>Overview</h1>
YeaZ-micromap (YeaZ-microscopy mapper) adapts the look of images from the target set so that they can be segmented using a neural network trained for the segmentation of images from the source set.
The tool consists of two modules: the style transfer module, which maps the look of images; and the YeaZ module, which segments the style-transferred images. As the style-transfer module, we used a generative adversarial network called CycleGAN. 

<p>

Paper: *Pan-microscope image segmentation based on a single training set*
<p align="center">
    
<img src="https://github.com/rahi-lab/YeaZ-micromap/assets/48595116/8ad9fb06-d23e-4afe-a34a-638251835131"/>

</p>



# Installation
### System requirements
The code was written in Python 3.9 and tested on Windows 11 and RedHat Linux server 7.7.

### Hardware requirements   

#### Minimum Requirements (for demo and prototyping)

- **Processor**: AMD Ryzen 5 5600H or equivalent
- **Memory (RAM)**: 16GB
- **Storage**: 100 GB available space
- **Graphics**: NVIDIA GeForce RTX 3060, 6GB VRAM

#### Recommended Requirements

- **Processor**: [TODO]
- **Memory (RAM)**: [TODO]
- **Storage**: [TODO]
- **Graphics**: [TODO]

### Installation steps

Installation time is less than 10 minutes.
</br>

 1. Clone the repository (```git clone https://github.com/rahi-lab/YeaZ-micromap```) or download it directly from the GitHub webpage
 2. Create a virtual environment ```conda create -n YeaZ-micromap python=3.9```
 3. Activate the virtual environment ```conda activate YeaZ-micromap```
 4. Install pytorch ```pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117```
 5. Navigate to the folder where you cloned the YeaZ-micromap repository and install required packages ```pip install -r requirements.txt```


# Usage

The code can be run from the command line and is split into two parts: (i) Training of the microscopy style-transfer using CycleGAN (ii) Evaluation of the training by segmenting the mapped images using a pre-trained YeaZ network for segmentation. 

More specifically:

1. *train_cyclegan.py* script performs:
    * style transfer training between the images in the trainA and trainB folders

2. *evaluate.py* script performs:
    *  style transfer on source dataset images in one of the specified folders (testA or testB) using the pretrained CycleGAN
    *  segmentation on the style-transferred images using the pretrained YeaZ weights, 
    *  evaluation of segmentation quality based on the segmented images and GT masks

    Both parts of the code rely on the following input data structure:
    ```
        input_data
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

3. Additonally, the helper function, *preprocessing.py*, prepares the raw input data, of variable sizes and contents, for style transfer training.


## Train CycleGAN
Training script arguments follow the established options nomenclature from the original cycleGAN repository (https://github.com/taesungp/contrastive-unpaired-translation). For more details see the comments in the code below.

### 1. Preprocessing

To preprocess the raw images into patches for style transfer training, use the following command on both source and target datasets:

```bash
$ python preprocess.py --src_path INPUT_FOLDER --dst_path OUTPUT_FOLDER
```
Please replace placeholders with actual values and descriptions relevant to your script.

#### Options

| Argument         | Description                                                | Default Value |
|------------------|------------------------------------------------------------|---------------|
| `--src_path`     | Path to the folder containing the images.                  | -             |
| `--dst_path`     | Path to the folder where patches will be saved.            | -             |
| `--var_thr`      | Empirical variance threshold for empty patches detection.  | `500000`      |
| `--scale_factor` | Factor to scale the images intensity by.                   | `1.0`         |
| `--patch_size`   | Size of the output patches.                                | `256`         |

### 2. Start a visdom server
```bash
$ python -m visdom.server
```
<p> Visdom is a visualization tool that communicates with the CycleGAN code during training and saves one example of mapping per epoch. This is useful for quickly checking whether the mapping qualitatively  makes sense. Saved data can be later accessed using an HTML interface, in Checkpoint/Experiment_Name/web/index.html</p>

### 3. Launch training

To initiate the training process, execute the following command:

```bash
$ python train_cyclegan.py \
    --dataroot GT_DATA_FOLDER \
    --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER \
    --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING \
    --grid_lambdas_A L1 L2 \
    --grid_lambdas_B L1
```
Please replace placeholders with actual values and descriptions relevant to your script.

#### Main Options

| Argument                        | Description                                                                 | Default Value |
|---------------------------------|-----------------------------------------------------------------------------|---------------|
| `--dataroot GT_DATA_FOLDER`     | Directory containing training images.                                       | -             |
| `--checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER` | Directory to save trained models. Models are saved after each epoch by default. | -             |
| `--name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING`            | Experiment name for future reference.                                        | -             |
| `--grid_lambdas_A L1 L2 ...`    | Cycle consistency loss weights for A->B->A mapping.                          | `10`          |
| `--grid_lambdas_B L1 L2 ...`    | Cycle consistency loss weights for B->A->B mapping.                          | `10`          |

If multiple lambda values are specified, a grid search will be performed.</br>
If no lambda values are specified, default values (10, 10) will be used.

#### Other Options

| Argument                | Description                                    | Default Value |
|-------------------------|------------------------------------------------|---------------|
| `--model cycle_gan`     | Generative model for transferring images.      | -             |
| `--gpu_ids GPU_ID`      | `-1` for CPU, `0` for GPU0, `0 1` for GPU0 and GPU1. | `0`           |
| `--batch_size BATCH_SIZE`| Batch size for training.                      | `1`           |
| `--n_epochs N_EPOCHS`   | Number of training epochs.                     | `200`         |
| `--n_epochs_decay N_EPOCHS_DECAY` | Epochs before linearly decaying the learning rate. | `200`         |
| `--lr LR`               | Initial learning rate for Adam optimizer.      | `0.0002`      |

## Evaluate the mapping using pretrained YeaZ
<p> For evaluating the segmentation accuracy, the user provides the directory with checkpoint weights from the CycleGAN training (<i>checkpoints_dir</i>), the DNN weights used for training of the source dataset (<i>path_to_yeaz_weights</i>), among other things. 

The rest of the arguments refer to either other trained CycleGAN specifications (_dataroot_, _name_, _model_) or to YeaZ segmentation (_threshold_, _min_seed_dist_, _min_epoch_, _max_epoch_, _epoch_step_). The dataroot folder contains the mask of the small annotated patch of the test image for only one of the domains (corresponding to the target set). If specified, a subpart (patch) of the big mask can be used for training evaluation instead of the whole mask. In that case _metrics_patch_borders_ should be supplied as an additional parameter. The resulting segmentation masks will be saved in _results_dir_ and the metrics of segmentation in _metrics_path_. </p>

To evaluate the style-transferred images and metrics, use the following command:
```bash
$ python evaluate.py \
    --dataroot GT_DATA_FOLDER \
    --checkpoints_dir GENERAL_CYCLE_GAN_TRAINING_FOLDER \
    --name NAME_OF_SPECIFIC_CYCLEGAN_TRAINING \
    --path_to_yeaz_weights PATH_TO_YEAZ_WEIGHTS \
    --min_epoch 1 \
    --max_epoch 201 \
    --epoch_step 5 \
    --results_dir RESULTS_FOLDER \
    --metrics_path METRICS_PATH
```
Please replace placeholders with actual values and descriptions relevant to your script.

#### Main Options

| Argument                        | Description                                          | Default Value |
|---------------------------------|------------------------------------------------------|---------------|
| `--dataroot`                    | Directory containing test images.                    | -             |
| `--checkpoints_dir`             | Directory with CycleGAN training checkpoints.        | -             |
| `--name`                        | Experiment name from CycleGAN training.              | -             |
| `--path_to_yeaz_weights`        | Path to the pretrained YeaZ weights.                 | -             |
| `--min_epoch`                   | First CycleGAN epoch for evaluation.                 | `1`           |
| `--max_epoch`                   | Last CycleGAN epoch for evaluation.                  | `201`         |
| `--epoch_step`                  | Evaluate every n-th epoch.                           | `5`           |
| `--results_dir`                 | Output folder for style-transferred images.          | -             |
| `--metrics_path`                | Path to save evaluation metrics (AP).                | -             |

#### Other Options

| Argument                       | Description                                           | Default Value |
|--------------------------------|-------------------------------------------------------|---------------|
| `--original_domain A or B`     | Target dataset to use test sets from.                | `A`           |
| `--skip_style_transfer`         | Skip style transfer if already performed.            | -             |
| `--skip_segmentation`           | Skip segmentation if already performed.              | -             |
| `--skip_metrics`                | Skip metrics if already evaluated.                   | -             |
| `--threshold`                   | Threshold used during YeaZ prediction.               | `0.5`         |
| `--min_seed_dist`               | Minimal seed distance between cells for YeaZ prediction.  | `5`           |
| `--metrics_patch_borders Y0 Y1 X0 X1` | Metrics patch borders, e.g., `480 736 620 876`.  | -             |
| `--plot_metrics`                | Plot evaluation metrics.                            | -             |


<h1>Demo</h1>

#### The demo showcases YeaZ-micromap capabilities for style transfer of yeast microscopy images from target to source domain, their segmentation in the source domain, and evaluation criteria (average precission) for selecting the best style transfer epoch for the following segmentation task.

Source domain: PhaseContrast</br>
Target domain: BrightField

Demo time: ~2-3h

1. Data download
    - Download the data from TODO
    - Upack the downloaded file and place its contents into _./data/_ folder

2. Data preprocessing
    - Preprocess PhaseContrast images: ```$ python preprocess.py --src_path ./data/input_data/PhaseContrast_demo/ --dst_path ./data/input_data/trainA/ --scale_factor 10```
    - Preprocess BrightField images: ```$ python preprocess.py --src_path ./data/input_data/BrightField_demo/ --dst_path ./data/input_data/trainB/```
    - Preprocessed PhaseContrast and BrightField images can be found in the folders _trainA_ and _trainB_ respectively (within the <i>./data/input_data/</i> folder)

3. Style transfer training
    - Start visdom: ```$ python -m visdom.server```
    - Run CycleGAN training: ```$ python train_cyclegan.py --dataroot ./data/input_data/ --name demo --checkpoints_dir ./data/checkpoints/ --gpu_ids 0 --n_epochs 100 --n_epochs_decay 0 --batch_size 1 --display_freq 1```
    - Track the training progress via visdom at http://localhost:8097/
    - All weights will be stored at _./data/checkpoints_ 

4. Evaluate domain adaptation
    - Run evaluate script: ```$ python evaluate.py --dataroot ./data/input_data/ --checkpoints_dir ./data/checkpoints/ --name demo_lambda_A_10.0_lambda_B_10.0 --path_to_yeaz_weights ./data/input_data/YeaZ_weights/weights_budding_PhC_multilab_0_1 --max_epoch 100 --results_dir ./data/results/ --metrics_path ./data/results/metrics_lambda_A_10.0_lambda_B_10.0.csv --metrics_patch_borders 200 456 200 456 --plot_metrics --original_domain B ```
    - You can find the style transfer output at <i>./data/results/demo_lambda_A_10.0_lambda_B_10.0/test_[EPOCH]/images/fake_A/wt_FOV9_PhC_absent.nd2_channel_10p.png</i> by replacing the EPOCH placeholder
    - You can find the generated segmentation masks from the style-transfered images at <i>./data/results/demo_lambda_A_10.0_lambda_B_10.0/test_[EPOCH]/images/fake_A/wt_FOV9_PhC_absent.nd2_channel_10p_mask.h5</i> by replacing the EPOCH placeholder.</br>
    You can utilze YeaZ (download from https://github.com/rahi-lab/YeaZ-GUI) to visualize the masks.
    - Average precision (AP) metrics can be found in the <i>./data/results/</i> folder, files: <i>metrics_lambda_A_10.0_lambda_B_10.0.csv and metrics_lambda_A_10.0_lambda_B_10.0.png</i>
