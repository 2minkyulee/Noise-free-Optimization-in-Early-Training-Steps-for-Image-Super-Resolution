import torch.nn.functional as F
import math
import os
import random
from typing import Any
import torchvision.transforms.functional as tvtf
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import random

# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`

def quantize(x):
    x = torch.clamp((x * 255.0).round(), 0, 255) / 255.
    x = x.clamp(0, 1)
    return x

def modcrop(img, n):
    assert type(n)==int or (int(n)-n)**2 < 0.0001
    
    h = (img.shape[-2]//n) * n
    w = (img.shape[-1]//n) * n
    
    return img[..., 0:h, 0:w]

def random_crop(x, patch_size, margin=0):
    """
    :param x: BCHW or CHW torch.tensor
    :param margin: doesnt crop from the region
    :return: BCHW or CHW torch.tensor
    """
    
    H, W = x.shape[-2:]
    h_start = random.randint(margin, H - patch_size - margin)
    w_start = random.randint(margin, W - patch_size - margin)
    x = _crop(x, h_start, w_start, patch_size, patch_size)
    
    return x

def random_crop_LRHRAUX(LR, HR, AUX, patch_size_HR, margin=0):
    """
    Random crop for dataset_LRHR + AUX. Identical to dataset_LRHR except for that it also returns a AUX.
    Provides aligned random crop.
    """
    H_LR, W_LR = LR.shape[-2:]
    H_HR, W_HR = HR.shape[-2:]
    assert AUX.shape == HR.shape
    
    scale_factor = W_HR // W_LR
    assert scale_factor == H_HR // H_LR  # assert equal ratio
    assert H_HR % scale_factor == W_HR % scale_factor == 0  # assert image size is multiple of scale_factor
    assert patch_size_HR % scale_factor == 0  # assert patch size is multiple of scale_factor
    
    h_start_HR = (random.randint(margin, H_HR - patch_size_HR - margin) // scale_factor) * scale_factor
    w_start_HR = (random.randint(margin, W_HR - patch_size_HR - margin) // scale_factor) * scale_factor
    HR = _crop(HR, h_start_HR, w_start_HR, patch_size_HR, patch_size_HR)
    AUX = _crop(AUX, h_start_HR, w_start_HR, patch_size_HR, patch_size_HR)
    
    h_start_LR = h_start_HR // scale_factor
    w_start_LR = w_start_HR // scale_factor
    patch_size_LR = patch_size_HR // scale_factor
    LR = _crop(LR, h_start_LR, w_start_LR, patch_size_LR, patch_size_LR)
    
    return LR, HR, AUX

def random_crop_LRHR(LR, HR, patch_size_HR, margin=0, other_LRs=[], other_HRs=[]):
    """
    Random crop for dataset_LRHR.
    Provides aligned random crop.
    """
    H_LR, W_LR = LR.shape[-2:]
    H_HR, W_HR = HR.shape[-2:]
    
    scale_factor = W_HR // W_LR
    assert scale_factor == H_HR // H_LR                         # assert equal ratio
    assert H_HR % scale_factor == W_HR % scale_factor == 0      # assert image size is multiple of scale_factor
    assert patch_size_HR % scale_factor == 0                    # assert patch size is multiple of scale_factor

    h_start_HR = (random.randint(margin, H_HR - patch_size_HR - margin) // scale_factor) * scale_factor
    w_start_HR = (random.randint(margin, W_HR - patch_size_HR - margin) // scale_factor) * scale_factor
    HR = _crop(HR, h_start_HR, w_start_HR, patch_size_HR, patch_size_HR)
    
    h_start_LR = h_start_HR // scale_factor
    w_start_LR = w_start_HR // scale_factor
    patch_size_LR = patch_size_HR // scale_factor
    LR = _crop(LR, h_start_LR, w_start_LR, patch_size_LR, patch_size_LR)


    if len(other_LRs)>0 and isinstance(other_LRs, list):
        assert len(other_LRs) == len(other_HRs)
        other_LRs_out = []
        other_HRs_out = []
        for other_LR in other_LRs:
            other_LRs_out.append(_crop(other_LR, h_start_LR, w_start_LR, patch_size_LR, patch_size_LR))
        for other_HR in other_HRs:
            other_HRs_out.append(_crop(other_HR, h_start_HR, w_start_HR, patch_size_HR, patch_size_HR))
    
        return LR, HR, other_LRs_out, other_HRs_out
    
    else:
        return LR, HR,


def random_hflip(x, p):
    if p > random.random():
        x = hflip(x)
    
    return x

def random_hflip_MultiImages(*images, p=0.5):
    """
    Random hflip for dataset_LRHR.
    Provides aligned hflip.
    """
    if p > random.random():
        images = tuple(map(hflip, images))
    return images
    
    # if p > random.random():
    #     LR = hflip(LR)
    #     HR = hflip(HR)
    #
    # return LR, HR


def random_vflip(x, p):
    if p > random.random():
        x = vflip(x)
    
    return x

def random_vflip_MultiImages(*images, p=0.5):
    """
    Random vflip for dataset_LRHR.
    Provides aligned vflip.
    """

    if p > random.random():
        images = tuple(map(vflip, images))
    return images

    # if p > random.random():
    #     LR = vflip(LR)
    #     HR = vflip(HR)
    #
    # return LR, HR


def random_rotate_MultiImages(*images, angles=(0, 90, 180, 270)):
    """
    Random rotate for dataset_LRHR.
    Provides aligned rotate.
    """
    angle = random.choice(angles)
    
    n_images = len(images)
    images = tuple(map(rotate, images, n_images*[angle]))
    return images
    
    
    
    # LR = rotate(LR, angle)
    # HR = rotate(HR, angle)
    # return LR, HR


def random_rotate(x, angles=(0, 90, 180, 270)):
    angle = random.choice(angles)
    x = rotate(x, angle)
    
    return x


def center_crop(x, patch_size):
    """
    :param x: BCHW or CHW torch.tensor
    :return: BCHW or CHW torch.tensor
    """
    patch_size = int(patch_size)
    H, W = x.shape[-2:]
    h_start = H//2 - patch_size//2
    w_start = W//2 - patch_size//2
    x = _crop(x, h_start, w_start, patch_size, patch_size)
    
    return x
    

def _crop(x, h_start, w_start, h, w):
    """
    :param x: BCHW or CHW torch.tensor
    :return: BCHW or CHW torch.tensor
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.dim() == 4
    
    x = x[:, :, h_start:h_start+h, w_start:w_start+w]
    x = x.squeeze(0)
    
    return x


def hflip(x):
    x = tvtf.hflip(x)
    return x


def vflip(x):
    x = tvtf.vflip(x)
    return x


def rotate(x, angle):
    x = tvtf.rotate(x, angle)
    return x


def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:
       weights, indices, sym_len_s, sym_len_e

    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale
    
    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)
    
    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)
    
    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)
    
    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2
    
    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)
    
    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    
    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)
    
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)
    
    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation

    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def bicubic_resize(image: Any, scale_factor: int, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (int): Scale factor. The same scale applies for both height and width.  <===== modified from original code
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.
        device (str): which device to use

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round

    """
    
    scale_factor = 1 / scale_factor  # <===== modified from original code
    
    
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True
    
    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4
    
    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)

    
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)
    
    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)
    
    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)
    
    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])
    
    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)
    
    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)
    
    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)
    
    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])
    
    
    out_2 = out_2.clamp(0, 1)
    
    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)
    
    return out_2


def padded_bicubic_resize(image, scale_factor, antialiasing=True):
    
    # n_pad = 4
    # image = F.pad(image, pad=4*[scale_factor*n_pad], mode="reflect")
    image = bicubic_resize(image=image, scale_factor=scale_factor, antialiasing=antialiasing)
    # image = image[:, n_pad:-n_pad, n_pad:-n_pad]
    
    return image
    

def BGRnumpy_to_RGBtensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR->RGB
    img = to_tensor(img)  # HWC -> CHW
    return img


def imsave(imgs, save_dir, names=None, verbose=True, round=True, jet=False):
    if names is None:
        names = ["tmp" + i for i in range(imgs.shape[0])]
    
    os.makedirs(save_dir, exist_ok=True)
    for img_idx in range(imgs.shape[0]):
        img = imgs[img_idx, :, :, :]
        name = names[img_idx]

        try:
            name = os.path.basename(name)
            name = os.path.join(save_dir, name)

            if round:
                img = quantize(img)
            
            img = to_pil_image(img)
            if verbose:
                print(f"saving {name}")
            img.save(name)
        
        
        
        
        except Exception as E:
            print(E)
