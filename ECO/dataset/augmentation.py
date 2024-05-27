from utils.img_proc_utils import random_crop_LRHR, random_hflip_MultiImages, random_vflip_MultiImages, random_rotate_MultiImages, random_rotate, random_hflip, random_vflip
import random

def basic_train_aug_MultiImages(*images):
    """
    Train Augmentation inside dataset.__getitem__().
    Identical to basic_train_aug but provides synchronized randomness between LR, HR.
    :param lr: lr image
    :param hr: hr image
    :return: augmented LRHR pair
    """
    
    images = random_rotate_MultiImages(*images)
    images = random_hflip_MultiImages(*images)
    images = random_vflip_MultiImages(*images)
    
    return images


def basic_train_aug(x):
    """
    Train Augmentation inside dataset.__getitem__()
    """
    
    x = random_rotate(x, angles=[0, 90, 180, 270])
    x = random_hflip(x, 0.5)
    x = random_vflip(x, 0.5)
    
    return x

