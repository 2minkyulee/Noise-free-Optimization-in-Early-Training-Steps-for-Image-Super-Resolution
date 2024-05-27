import time
from utils.misc_utils import isNullStr, path_exists
import os
import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import importlib
from utils.misc_utils import metric_init
from dataset.syntheticSR_dataset_preprocessed import Dataset_LRHR, Dataset_onlyLR, Dataset_LRHRAUX,\
    Dataset_LRHR_Mixup, Dataset_LRHR_AlphaSample, Dataset_EcooPlus
from dataset.realesrgan_dataset import Dataset_LRHR_RealESRGAN



def define_optimizer(model, config):

    if config.optimizer == "Adam":
        module = importlib.import_module("torch.optim.adam")
        optimizer_attr = getattr(module, "Adam")
        optim = optimizer_attr(
            model.parameters(),
            **config.optimizer_kwargs
        )
    elif config.optimizer == "AdamW":
        module = importlib.import_module("torch.optim.adamw")
        optimizer_attr = getattr(module, "AdamW")
        optim = optimizer_attr(
            model.parameters(),
            **config.optimizer_kwargs
        )
    elif config.optimizer == "Lamb":
        module = importlib.import_module("utils.optimizer_utils")
        optimizer_attr = getattr(module, "Lamb")
        optim = optimizer_attr(
            model.parameters(),
            **config.optimizer_kwargs
        )
    elif config.optimizer == "SGD":
        module = importlib.import_module("torch.optim.sgd")
        optimizer_attr = getattr(module, "SGD")
        optim = optimizer_attr(
            model.parameters(),
            **config.optimizer_kwargs
        )
    


    else:
        raise NotImplementedError
    
    
    return optim


def define_scheduler(optimizer, config):


    if config.scheduler in ["StepLR", "MultiStepLR"]:
        module = importlib.import_module("torch.optim.lr_scheduler")
        sched_attr = getattr(module, config.scheduler)
        sched = sched_attr(
            optimizer=optimizer,
            **config.scheduler_kwargs
        )
    
    elif config.scheduler == "WarmupCosineLR":
        module = importlib.import_module("utils.lr_scheduler_utils")
        sched_attr = getattr(module, config.scheduler)
        sched = sched_attr(
            optimizer=optimizer,
            **config.scheduler_kwargs
        )
    
    else:
        raise NotImplementedError
    
    return sched





class TrainFlags():
    def __init__(self, curr_iter, max_iter, metrics):
        self.STOP = False
        self.set_pdb = False
        
        self.best_metric = {metric:metric_init(metric) for metric in metrics}
        
        self.timer = Timer()
        self.curr_iter = curr_iter
        self.max_iter = max_iter
        self.tic_iter = 0
        self.toc_iter = 0
        
        self.last_loss = 1e2
        
    def step(self):
        self.curr_iter += 1
        if self.curr_iter >= self.max_iter:
            self.STOP = True
            
    def tic(self):
        self.timer.tic()
        self.tic_iter = self.curr_iter

    def toc(self):
        self.timer.toc()
        self.toc_iter = self.curr_iter

    def print_times(self):
        curr_time = time.time()
        sec_per_iter = (self.timer.toc_time - self.timer.tic_time) / (self.toc_iter - self.tic_iter)
        left_iter_todo = self.max_iter - self.curr_iter
        eta = sec_per_iter * left_iter_todo
        print(
            f"| {self.curr_iter:>10d}/{self.max_iter:<10d}  iter"
            f"| {sec_per_iter:3.3f} sec per iter "
            f"| ETA {self.timer.norm_time(eta):21s} "
        )
    
    
    
class Timer():
    
        def __init__(self):
            self.tic_time = None
            self.toc_time = None
            
        def tic(self):
            self.tic_time = time.time()
        
        def toc(self):
            self.toc_time = time.time()
            return self.toc_time
            
        def norm_time(self, sec):
            sec_leftover = sec
            H = int(sec_leftover // 3600)
    
            sec_leftover = sec - 3600 * H
            M = int(sec_leftover // 60)
    
            sec_leftover = sec - 3600 * H - 60 * M
            S = int(sec_leftover)
    
            return f"{H} hours {M} min {S} sec"




def define_loader(mode, config):
    
    path_imageLR = getattr(config, f"{mode}_path_imageLR")
    path_imageHR = getattr(config, f"{mode}_path_imageHR")
    path_imageAUX = getattr(config, f"{mode}_path_imageAUX")
    
    path_imageLR2 = getattr(config, f"{mode}_path_imageLR2") if hasattr(config, f"{mode}_path_imageLR2") else ""
    path_imageHR2 = getattr(config, f"{mode}_path_imageHR2") if hasattr(config, f"{mode}_path_imageHR2") else ""
    
    
    
    
    batch_size = getattr(config, f"{mode}_batch_size")
    patch_size = getattr(config, f"{mode}_patch_size")
    scale_factor = getattr(config, f"{mode}_scale_factor")
    
    
    # get dataset_module type
    if isNullStr(path_imageLR) and isNullStr(path_imageHR):
        raise RuntimeError
    
    elif config.trainer == "RealESRGAN_Trainer":
        dataset_module = Dataset_LRHR_RealESRGAN
    
    elif not isNullStr(path_imageAUX):   #todo.
        print(f"loading dataloader type Dataset_LRHRAUX for {mode}")
        dataset_module = Dataset_LRHRAUX
    
    elif not isNullStr(path_imageLR) and isNullStr(path_imageHR):
        print(f"loading dataloader type Dataset_onlyLR for {mode}")
        dataset_module = Dataset_onlyLR
    
    # elif isNullStr(path_imageLR) and not isNullStr(path_imageHR):
    #     dataset_module = Dataset_onlyHR
    #     print(f"loading dataloader type Dataset_onlyHR for {mode}")
    
    elif (not isNullStr(path_imageLR) and not isNullStr(path_imageHR))\
            and (isNullStr(path_imageLR2) and (isNullStr(path_imageHR2))):
        dataset_module = Dataset_LRHR
        print(f"loading dataloader type Dataset_LRHR for {mode}")
    
    elif (not isNullStr(path_imageLR) and not isNullStr(path_imageHR))\
        and (not isNullStr(path_imageLR2) and not isNullStr(path_imageHR2)):
        
        if config.trainer == "EcooPlusTrainer":
            dataset_module = Dataset_EcooPlus
        else:
            if config.alpha_sample:
                dataset_module = Dataset_LRHR_AlphaSample
                print(f"loading dataloader type Dataset_LRHR_AlphaSample for {mode}")
            elif (config.mixup_shift==0) or bool(config.mixup_shift):
                dataset_module = Dataset_LRHR_Mixup
                print(f"loading dataloader type Dataset_LRHR_mixup for {mode}")
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    if mode == "train":
        assert path_exists(path_imageHR), f"{path_imageHR} is not a valid path."
        
        dataset = dataset_module(
            path_image_HR=path_imageHR,
            path_image_LR=path_imageLR,
            path_image_HR2=path_imageHR2,
            path_image_LR2=path_imageLR2,
            path_image_AUX=path_imageAUX,  #todo
            patch_size=patch_size,
            crop="random",
            scale_factor=scale_factor,
            train_aug=config.train_aug,
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size= batch_size,
            num_workers=0 if config.debug else config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
            shuffle=config.shuffle_train_data
        )

    elif mode == "valid":
        
        assert path_exists(path_imageHR), f"{path_imageHR} is not a valid path."
        
        dataset = dataset_module(
            path_image_HR=path_imageHR,
            path_image_LR=path_imageLR,
            patch_size=patch_size,
            crop="center" if patch_size>0 else None,
            scale_factor=scale_factor,
            train_aug=None,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1 if config.debug else batch_size,
            num_workers=0 if config.debug else config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            shuffle=False
        )

    elif mode == "test":
        
        assert path_exists(path_imageHR) or path_exists(path_imageLR)
        
        dataset = dataset_module(
            path_image_HR=path_imageHR,
            path_image_LR=path_imageLR,
            patch_size=patch_size,
            crop="center" if patch_size>0 else False,
            scale_factor=scale_factor,
            train_aug=None,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1 if config.debug else batch_size,
            num_workers=0 if config.debug else config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            shuffle=False
        )

    else:
        raise NotImplementedError(f"Expected loader mode train|valid|test but got {mode}")




    return dataloader


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


