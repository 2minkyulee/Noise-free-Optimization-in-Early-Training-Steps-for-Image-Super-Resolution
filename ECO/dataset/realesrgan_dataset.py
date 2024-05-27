"""
Modified from BasicSR
"""


import os
from PIL import Image
from dataset.augmentation import basic_train_aug_MultiImages, basic_train_aug
from utils.img_proc_utils import bicubic_resize, random_crop, center_crop, random_crop_LRHR, quantize, random_crop_LRHRAUX, modcrop
from torchvision.transforms.functional import to_tensor, rotate
from dataset.syntheticSR_dataset_preprocessed import DefaultDataSet
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


class Dataset_LRHR_RealESRGAN(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, patch_size, crop, scale_factor, train_aug, **kwargs):
        
        super(Dataset_LRHR_RealESRGAN, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )
        assert len(self.image_names_HR) == len(self.image_names_LR), print(len(self.image_names_HR), len(self.image_names_LR))

    
        opt_ker = {
            # first deg
            "blur_kernel_size": 21,
            "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            "sinc_prob": 0.1,
            "blur_sigma": [0.2, 3],
            "betag_range": [0.5, 4],
            "betap_range": [1, 2],
            
            # second deg
            "blur_kernel_size2": 21,
            "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            "sinc_prob2": 0.1,
            "blur_sigma2": [0.2, 1.5],
            "betag_range2": [0.5, 4],
            "betap_range2": [1, 2],
    
            # final sinc
            "final_sinc_prob": 0.8,
        }
    
    
    
        # blur settings for the first degradation
        self.blur_kernel_size = opt_ker['blur_kernel_size']
        self.kernel_list = opt_ker['kernel_list']
        self.kernel_prob = opt_ker['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt_ker['blur_sigma']
        self.betag_range = opt_ker['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt_ker['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt_ker['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt_ker['blur_kernel_size2']
        self.kernel_list2 = opt_ker['kernel_list2']
        self.kernel_prob2 = opt_ker['kernel_prob2']
        self.blur_sigma2 = opt_ker['blur_sigma2']
        self.betag_range2 = opt_ker['betag_range2']
        self.betap_range2 = opt_ker['betap_range2']
        self.sinc_prob2 = opt_ker['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt_ker['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
    
    
    
    
    
    def __getitem__(self, idx):
        
        
        
        #  --------------------- open image --------------------------- #
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)
        
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
        
        if image_HR.shape[0] == 1:
            image_HR = image_HR.repeat(3, 1, 1)  # for set14 (or other grayscale)
            image_LR = image_LR.repeat(3, 1, 1)

        #  --------------------- basic train aug --------------------------- #
        
        self.assert_shape(LR=image_LR, HR=image_HR)
        assert os.path.basename(self.image_names_LR[idx]) == os.path.basename(self.image_names_HR[idx])
        
        # apply crop and apply train aug on LR, HR
        if self.crop == "random" and self.patch_size > 0:
            image_LR, image_HR = random_crop_LRHR(image_LR, image_HR, self.patch_size)
        
        elif self.crop == "center" and self.patch_size > 0:
            image_LR = center_crop(image_LR, self.patch_size // self.scale_factor)
            image_HR = center_crop(image_HR, self.patch_size)
        else:
            assert not self.patch_size > 0
        
        if self.train_aug:
            image_LR, image_HR = basic_train_aug_MultiImages(image_LR, image_HR)
        
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        
        
        return {
            "LR": image_LR.clamp(0, 1),
            "HR": image_HR.clamp(0, 1),
            "name": self.image_names_LR[idx],
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }