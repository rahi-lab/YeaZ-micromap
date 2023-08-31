import os.path
from data.base_dataset import BaseDataset, get_transform, get_convert_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import torch


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if os.path.exists(self.dir_A):
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        else:
            self.A_paths = []
        
        if os.path.exists(self.dir_B):
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        else:
            self.B_paths = []
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if self.A_size == 0 and self.B_size == 0:
            raise ValueError("A_size and B_size are both 0")

        if opt.phase == "train" and any([self.A_size == 0, self.B_size == 0]):
            raise ValueError("A_size or B_size is 0 during training")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if self.A_size:
            if self.opt.serial_batches:   # make sure index is within the range
                index_A = index % self.A_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_A = random.randint(0, self.A_size - 1)
            A_path = self.A_paths[index_A]
            A_img = Image.open(A_path)
            if np.array(A_img).dtype.name == 'uint16': #dtype -> dtype.name
                A_img = np.uint8(np.array(A_img) / 256)
                A_img = Image.fromarray(A_img)
        else:
            A_img = Image.new('RGB', (self.opt.load_size, self.opt.load_size), (0, 0, 0))
            A_path = 'placeholder.png'

        if self.B_size:
            index_B = index
            B_path = self.B_paths[index_B % self.B_size]  # make sure index is within the range
            B_img = Image.open(B_path)
            if np.array(B_img).dtype.name == 'uint16': #dtype -> dtype.name
                B_img = np.uint8(np.array(B_img) / 256)
                B_img = Image.fromarray(B_img)
        else:
            B_img = Image.new('RGB', (self.opt.load_size, self.opt.load_size), (0, 0, 0))
            B_path = 'placeholder.png'

        

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        grayscale = (self.opt.input_nc == 1)
        transform = get_transform(modified_opt, grayscale=grayscale, convert=False)
        convert_transform = get_convert_transform(grayscale=grayscale)

        if grayscale:
            A_img = A_img.convert('LA')
            B_img = B_img.convert('LA')
        else:
            A_img = A_img.convert('RGB')
            B_img = B_img.convert('RGB')

        B = convert_transform(transform(B_img))
        A = convert_transform(transform(A_img))
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
