MODEL_NAME="cut"
IMG_SIZE="286"              # set to the size of your images (or approximation if it is not fixed)

python train.py \
  --load_size ${IMG_SIZE} \
  --crop_size ${IMG_SIZE} \
  --input_nc 1 \            # number of channels : 1 for grayscale, 3 for RGB
  --output_nc 1 \           # number of channels : 1 for grayscale, 3 for RGB
  --name ${MODEL_NAME} \
  --save_epoch_freq 50 \
  --lambda_GAN 1.0 \        # weight of the loss of GAN (related to style)
  --lambda_NCE 10.0 \       # weight of the contrastive loss (related to content)
  --n_epochs 300 \          # number of epochs with fixed learning rate
  --n_epochs_decay 100      # number of epochs with learning rate decay (so total number of epochs is the sum)

python test.py \
  --load_size ${IMG_SIZE} \
  --crop_size ${IMG_SIZE} \
  --input_nc 1 \
  --output_nc 1 \
  --name ${MODEL_NAME} \
  --phase train \
  --lambda_GAN 1.0 \
  --lambda_NCE 10.0 
