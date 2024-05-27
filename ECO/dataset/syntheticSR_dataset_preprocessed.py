import os
from utils.misc_utils import isNullStr
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataset.augmentation import basic_train_aug_MultiImages, basic_train_aug
from utils.img_proc_utils import bicubic_resize, random_crop, center_crop, random_crop_LRHR, quantize, random_crop_LRHRAUX, modcrop
from utils.misc_utils import get_image_names, clear_path

import torch
from torchvision.transforms.functional import to_tensor, rotate


class DefaultDataSet(Dataset):  # dont used this directly
    def __init__(self,
                 path_image_HR=None,
                 path_image_LR=None,
                 path_image_AUX=None,
                 patch_size=None,
                 crop=None,
                 scale_factor=None,
                 train_aug=None,
                 ):
        
        # basic
        self.train_aug = train_aug
        self.patch_size = patch_size
        self.crop = crop
        self.scale_factor = scale_factor
        self.image_names_HR = get_image_names(path_image_HR)
        self.image_names_LR = get_image_names(path_image_LR)
        self.image_names_AUX = get_image_names(path_image_AUX)
        self.mixup_alpha = 1  # (a)*imageHR + (1-a)*imageHR2
        
        
        

    def assert_shape(self, LR, HR):
        """
        asserts that LR and HR has image only scale difference (but ignore up-to mod12)
        """
        assert (LR.shape[-1] * self.scale_factor) // 12 == HR.shape[-1] // 12,\
            f"LR: {LR.shape}, {LR.shape[-1] * self.scale_factor // 12}, HR: {HR.shape}, {HR.shape[-1] // 12}"
        
        assert (LR.shape[-2] * self.scale_factor) // 12 == HR.shape[-2] // 12,\
            f"LR: {LR.shape[-2] * self.scale_factor // 12}, HR: {HR.shape[-2] // 12}"
    
    def __getitem__(self, item):
        pass
    
    def __len__(self):
        return len(self.image_names_HR)


# @tag [Define Dataset Train]
class Dataset_onlyLR(DefaultDataSet):
    """
    Dataset only given LR folder.
    Used for testing. (generally for real-world dataset, where GT image does not exist)
    """

    def __init__(self, path_image_HR, path_image_LR, patch_size, crop, scale_factor, train_aug, **kwargs):
        super(Dataset_onlyLR, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )
        assert isNullStr(path_image_HR)
    
    def __len__(self):
        return len(self.image_names_LR)
    
    def __getitem__(self, idx):

        # read
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)

        # apply crop and apply train aug only on LR
        if self.crop == "random" and self.patch_size > 0:
            image_LR = random_crop(image_LR, self.patch_size)
        elif self.crop == "center" and self.patch_size > 0:
            image_LR = center_crop(image_LR, self.patch_size)
        else:
            assert not self.patch_size > 0
            
        c, h, w = image_LR.shape
        image_HR = torch.zeros(c, h*self.scale_factor, w*self.scale_factor)

        if self.train_aug:
            image_LR = basic_train_aug(image_LR)

        self.assert_shape(LR=image_LR, HR=image_HR)
        
        return {
            "LR": image_LR.clamp(0, 1),
            "HR": image_HR.clamp(0, 1),
            "name": self.image_names_LR[idx]
        }


# @tag [Define Dataset Train]
class Dataset_LRHR(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, patch_size, crop, scale_factor, train_aug, **kwargs):
    
        super(Dataset_LRHR, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )
        assert len(self.image_names_HR) == len(self.image_names_LR), print(len(self.image_names_HR), len(self.image_names_LR))
    
    def __getitem__(self, idx):
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)

        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
        
        if image_HR.shape[0] == 1:
            image_HR = image_HR.repeat(3, 1, 1)  # for set14 (or other grayscale)
            image_LR = image_LR.repeat(3, 1, 1)
        

        self.assert_shape(LR=image_LR, HR=image_HR)
        assert os.path.basename(self.image_names_LR[idx])==os.path.basename(self.image_names_HR[idx])
        
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
        
        return {
            "LR": image_LR.clamp(0, 1),
            "HR": image_HR.clamp(0, 1),
            "name": self.image_names_LR[idx]
        }


class Dataset_EcooPlus(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, path_image_HR2, path_image_LR2, patch_size, crop, scale_factor, train_aug, **kwargs):
        super(Dataset_EcooPlus, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )
        
        self.image_names_HR2 = get_image_names(path_image_HR2)
        self.image_names_LR2 = get_image_names(path_image_LR2)
        
        assert len(self.image_names_HR) == len(self.image_names_LR), \
            print(path_image_HR, len(self.image_names_HR), path_image_LR, len(self.image_names_LR))
    
    def __getitem__(self, idx):
        
        assert os.path.basename(self.image_names_LR[idx]) == os.path.basename(self.image_names_HR[idx])
        assert os.path.basename(self.image_names_LR2[idx]) == os.path.basename(self.image_names_HR2[idx])
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)
    
        image_HR2 = Image.open(self.image_names_HR2[idx])
        image_HR2 = to_tensor(image_HR2)
    
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
    
        image_LR2 = Image.open(self.image_names_LR2[idx])
        image_LR2 = to_tensor(image_LR2)

        if self.crop == "random" and self.patch_size > 0:
            image_LR, image_HR, other_LRs, other_HRs = random_crop_LRHR(
                image_LR,
                image_HR,
                other_LRs=[image_LR2],
                other_HRs=[image_HR2],
                patch_size_HR=self.patch_size,
            )
        else:
            raise NotImplementedError
        
        image_LR2 = other_LRs[0]
        image_HR2 = other_HRs[0]
        image_LR, image_LR2, image_HR, image_HR2 = basic_train_aug_MultiImages(image_LR, image_LR2, image_HR, image_HR2)
        
        # do not clamp!! This is used to train linearity.
        # While training this, it subtracts and can cause negative values.
        # Need to keep consistency with training results and theoretical equations.
        
        ret = {
            "x_SRdown": image_LR,
            "y_SR": image_HR,
            "x_HRdown": image_LR2,
            "y_HR": image_HR2,
            "name": self.image_names_LR2[idx]
        }
        
        return ret
        
        
class Dataset_LRHR_Mixup(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, path_image_HR2, path_image_LR2, patch_size, crop, scale_factor, train_aug, **kwargs):
        
        super(Dataset_LRHR_Mixup, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )

        
        self.image_names_HR2 = get_image_names(path_image_HR2)
        self.image_names_LR2 = get_image_names(path_image_LR2)
        
        assert len(self.image_names_HR) == len(self.image_names_LR), \
            print(path_image_HR, len(self.image_names_HR), path_image_LR, len(self.image_names_LR))
            
    
    def __getitem__(self, idx):
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)
        
        image_HR2 = Image.open(self.image_names_HR2[idx])
        image_HR2 = to_tensor(image_HR2)
        
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
        
        image_LR2 = Image.open(self.image_names_LR2[idx])
        image_LR2 = to_tensor(image_LR2)
        
        a = self.mixup_alpha
        image_HR_mixup = a * image_HR + (1 - a) * image_HR2
        image_LR_mixup = a * image_LR + (1 - a) * image_LR2
        
        if image_HR_mixup.shape[0] == 1:
            image_HR_mixup = image_HR_mixup.repeat(3, 1, 1)  # for set14 (or other grayscale)
            image_LR_mixup = image_LR_mixup.repeat(3, 1, 1)
        
        self.assert_shape(LR=image_LR_mixup, HR=image_HR_mixup)
        assert os.path.basename(self.image_names_LR[idx]) == os.path.basename(self.image_names_HR[idx])
        assert os.path.basename(self.image_names_LR2[idx]) == os.path.basename(self.image_names_HR2[idx])
        
        # apply crop and apply train aug on LR, HR
        if self.crop == "random" and self.patch_size > 0:
            image_LR_mixup, image_HR_mixup = random_crop_LRHR(image_LR_mixup, image_HR_mixup, self.patch_size)
        
        elif self.crop == "center" and self.patch_size > 0:
            image_LR_mixup = center_crop(image_LR_mixup, self.patch_size // self.scale_factor)
            image_HR_mixup = center_crop(image_HR_mixup, self.patch_size)
        else:
            assert not self.patch_size > 0
        
        if self.train_aug:
            image_LR_mixup, image_HR_mixup = basic_train_aug_MultiImages(image_LR_mixup, image_HR_mixup)
        
        return {
            "LR": image_LR_mixup.clamp(0, 1),
            "HR": image_HR_mixup.clamp(0, 1),
            "name": self.image_names_LR2[idx]
        }


class Dataset_LRHR_AlphaSample(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, path_image_HR2, path_image_LR2, patch_size, crop, scale_factor, train_aug, **kwargs):
        
        super(Dataset_LRHR_AlphaSample, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug,
        )
        
        self.image_names_HR2 = get_image_names(path_image_HR2)
        self.image_names_LR2 = get_image_names(path_image_LR2)
        
        assert len(self.image_names_HR) == len(self.image_names_LR), \
            print(path_image_HR, len(self.image_names_HR), path_image_LR, len(self.image_names_LR))
    
    def __getitem__(self, idx):
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)
        
        image_HR2 = Image.open(self.image_names_HR2[idx])
        image_HR2 = to_tensor(image_HR2)
        
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
        
        image_LR2 = Image.open(self.image_names_LR2[idx])
        image_LR2 = to_tensor(image_LR2)
        
        a = self.mixup_alpha
        if a > torch.rand(1):
            image_HR_mixup = image_HR
            image_LR_mixup = image_LR
        else:
            image_HR_mixup = image_HR2
            image_LR_mixup = image_LR2
        
        
        if image_HR_mixup.shape[0] == 1:
            image_HR_mixup = image_HR_mixup.repeat(3, 1, 1)  # for set14 (or other grayscale)
            image_LR_mixup = image_LR_mixup.repeat(3, 1, 1)
        
        self.assert_shape(LR=image_LR_mixup, HR=image_HR_mixup)
        assert os.path.basename(self.image_names_LR[idx]) == os.path.basename(self.image_names_HR[idx])
        assert os.path.basename(self.image_names_LR2[idx]) == os.path.basename(self.image_names_HR2[idx])
        
        # apply crop and apply train aug on LR, HR
        if self.crop == "random" and self.patch_size > 0:
            image_LR_mixup, image_HR_mixup = random_crop_LRHR(image_LR_mixup, image_HR_mixup, self.patch_size)
        
        elif self.crop == "center" and self.patch_size > 0:
            image_LR_mixup = center_crop(image_LR_mixup, self.patch_size // self.scale_factor)
            image_HR_mixup = center_crop(image_HR_mixup, self.patch_size)
        else:
            assert not self.patch_size > 0
        
        if self.train_aug:
            image_LR_mixup, image_HR_mixup = basic_train_aug_MultiImages(image_LR_mixup, image_HR_mixup)
        
        return {
            "LR": image_LR_mixup.clamp(0, 1),
            "HR": image_HR_mixup.clamp(0, 1),
            "name": self.image_names_LR2[idx]
        }


# @tag [Define Dataset Train]
class Dataset_LRHRAUX(DefaultDataSet):
    def __init__(self, path_image_HR, path_image_LR, path_image_AUX, patch_size, crop, scale_factor, train_aug, **kwargs):
    
        super(Dataset_LRHRAUX, self).__init__(
            path_image_HR=path_image_HR,
            path_image_LR=path_image_LR,
            path_image_AUX=path_image_AUX,
            patch_size=patch_size,
            crop=crop,
            scale_factor=scale_factor,
            train_aug=train_aug
        )
        assert len(self.image_names_HR) == len(self.image_names_LR), print(len(self.image_names_HR), len(self.image_names_LR))
    
    def __getitem__(self, idx):
        
        image_HR = Image.open(self.image_names_HR[idx])
        image_HR = to_tensor(image_HR)
        if image_HR.shape[0] == 1:
            image_HR = image_HR.repeat(3, 1, 1)  # for set14 (or other grayscale)

        image_AUX = Image.open(self.image_names_AUX[idx])
        image_AUX = to_tensor(image_AUX)
        
        image_LR = Image.open(self.image_names_LR[idx])
        image_LR = to_tensor(image_LR)
        
        self.assert_shape(LR=image_LR, HR=image_HR)
        
        # apply crop and apply train aug on LR, HR
        if self.crop == "random" and self.patch_size > 0:
            image_LR, image_HR, image_AUX = random_crop_LRHRAUX(image_LR, image_HR, image_AUX, self.patch_size)
        
        elif self.crop == "center" and self.patch_size > 0:
            image_LR = center_crop(image_LR, self.patch_size // self.scale_factor)
            image_HR = center_crop(image_HR, self.patch_size)
            image_AUX = center_crop(image_AUX, self.patch_size)
        else:
            assert not self.patch_size > 0
        
        if self.train_aug:
            image_LR, image_HR, image_AUX = basic_train_aug_MultiImages(image_LR, image_HR, image_AUX)
        
        return {
            "LR": image_LR.clamp(0, 1),
            "HR": image_HR.clamp(0, 1),
            "AUX": image_AUX.clamp(0, 1),
            "name": self.image_names_LR[idx]
        }
