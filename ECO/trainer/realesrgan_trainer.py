'''
Code modified from
https://github.com/xinntao/Real-ESRGAN/
'''


import importlib
import copy
from torch.nn import DataParallel as DP
from torchvision.utils import make_grid
from PIL import Image
import wandb
from loss import perceptual_loss, adversarial_loss, reconstruction_loss
from loss import LossLogger
import os
from torchvision.transforms.functional import to_pil_image, to_tensor
from utils.img_proc_utils import modcrop
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode
from trainer.trainer import DefaultTrainer
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
import random



class Gradient_Net(nn.Module):
    
    def __init__(self, device, pad=0):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.pad = pad
    
    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=self.pad)
        grad_y = F.conv2d(x, self.weight_y, padding=self.pad)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

class RealESRGAN_Trainer(DefaultTrainer):
    """
    Trainer for Real-ESRGAN.
    Two main points of the implementation is as below.
    1) GPU accelerated batch-wise second-order degradation
    2) Training Pool to increase diversity, limited by batch-wise augmentation
    """
    def __init__(self, *args, **kwargs):
        super(RealESRGAN_Trainer, self).__init__(*args, **kwargs)
        
        self._initialize_training_pool(max_pool_size=self.config.training_pool_size, tolerance=self.config.training_pool_size_tolerance)
        self.grad_net = Gradient_Net('cuda')
        self.down_fn = Resize(
            (self.config.train_patch_size // self.config.train_scale_factor, self.config.train_patch_size // self.config.train_scale_factor),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )
        self.up_fn = Resize(
            (self.config.train_patch_size, self.config.train_patch_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )
        
        # losses
        self.loss_logger = LossLogger(wandb_run=self.wandb_run, train_flags=self.train_flags)
        self.perceptual_loss = perceptual_loss.PerceptualLoss(loss_logger=self.loss_logger).cuda()
        self.adv_loss = adversarial_loss.AdversarialLoss(loss_logger=self.loss_logger, discriminator_type="UNetSN", wandb_run=self.wandb_run, relative=False).cuda()
        self.pixel_loss = reconstruction_loss.L1_loss(config=None, loss_logger=self.loss_logger).cuda()
        self.loss_coef = {
            "pixel": 1.,
            "percep": 1.,
            "adv": 0.1,
        }
        
        
        
        # degradations
        self.pool_device = "cuda"
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        
        self.deg_opt = {
            # scale
            "scale": self.config.train_scale_factor,
            
            # USM the ground-truth
            "l1_gt_usm": True,
            "percep_gt_usm": True,
            "gan_gt_usm": False,
    
            # the first degradation process
            "resize_prob": [0.2, 0.7, 0.1],  # up, down, keep,
            "resize_range": [0.15, 1.5],
            "gaussian_noise_prob": 0.5,
            "noise_range": [1, 30],
            "poisson_scale_range": [0.05, 3],
            "gray_noise_prob": 0.4,
            "jpeg_range": [30, 95],
    
            # the second degradation process
            "second_blur_prob": 0.8,
            "resize_prob2": [0.3, 0.4, 0.3],  # up, down, keep,
            "resize_range2": [0.3, 1.2],
            "gaussian_noise_prob2": 0.5,
            "noise_range2": [1, 25],
            "poisson_scale_range2": [0.05, 2.5],
            "gray_noise_prob2": 0.4,
            "jpeg_range2": [30, 95],
        }

    def ent_to_psnr(self, x):
    
        gray_x = torch.mean(x, dim=1, keepdim=True)
        eps = 1e-8
    
        bic_recon = self.up_fn(self.down_fn(gray_x)).clamp(0, 1)
        diff = bic_recon - gray_x
        mse = torch.mean(diff.pow(2), dim=(-1, -2, -3), keepdim=True)
        psnr = -10 * torch.log10(mse + eps)
    
        grad_x = self.grad_net(gray_x)
        norm_grad_x = (grad_x / (torch.sum(grad_x, dim=(-1, -2), keepdim=True))).abs()
    
        ent = -1 * torch.sum(norm_grad_x * torch.log(norm_grad_x + eps), dim=(-1, -2), keepdim=True)
        e2p = ent / psnr
    
        return e2p

    # -------------------- training pool --------------------
    @torch.no_grad()
    def _initialize_training_pool(self, max_pool_size, tolerance):
        """
        :param max_pool_size: maximum pool size
        :param tolerance: tolerance to exceed maximum pool size (thus, actual pool size= poolsize+tolerace)
        :return:
        """
        hr_size = self.config.train_patch_size
        lr_size = self.config.train_patch_size // self.config.train_scale_factor
        
        
        self.training_pool = {
            "LR": torch.zeros(max_pool_size+tolerance, 3, lr_size, lr_size).cuda(),
            "LR_deg": torch.zeros(max_pool_size+tolerance, 3, lr_size, lr_size).cuda(),
            "HR": torch.zeros(max_pool_size+tolerance, 3, hr_size, hr_size).cuda(),
            "HR_usm": torch.zeros(max_pool_size+tolerance, 3, hr_size, hr_size).cuda(),
            "is_full": False,
            "curr_n_samples": 0,                      # how many samples currently in training pool. Acts as pointer and to check if pool is full.
            "max_n_samples": max_pool_size            # how many samples at max, in training pool
        }

    @torch.no_grad()
    def enqueue_training_pool(self, data_batch):
        
        
        # queue
        bs = data_batch["LR"].shape[0]
        
        ptr = self.training_pool["curr_n_samples"]  # idx pointer. Add at the end.
        
        self.training_pool["LR"][ptr:ptr+bs] = data_batch["LR"].clone().to(self.pool_device, non_blocking=True)
        self.training_pool["HR"][ptr:ptr+bs] = data_batch["HR"].clone().to(self.pool_device, non_blocking=True)
        self.training_pool["LR_deg"][ptr:ptr+bs] = data_batch["LR_deg"].clone().to(self.pool_device, non_blocking=True)
        self.training_pool["HR_usm"][ptr:ptr+bs] = data_batch["HR_usm"].clone().to(self.pool_device, non_blocking=True)
        
        self.training_pool["curr_n_samples"] += bs  # increase pointer
        
        # remove if over max pool size
        if self.training_pool["curr_n_samples"] > self.training_pool["max_n_samples"]:
            max_n = self.training_pool["max_n_samples"]
            curr_n = self.training_pool["curr_n_samples"]

            # select random samples only up to max pool size
            print("Exceed max pool size! Removing...")
            idx = torch.randperm(curr_n)[:max_n]
            self.training_pool["LR"] = self.training_pool["LR"][idx]
            self.training_pool["HR"] = self.training_pool["HR"][idx]
            self.training_pool["LR_deg"] = self.training_pool["LR_deg"][idx]
            self.training_pool["HR_usm"] = self.training_pool["HR_usm"][idx]
            self.training_pool["curr_n_samples"] = self.training_pool["max_n_samples"]  # set pointer. dont actually remove it
        
        
        # if full, set flag as True
        if self.training_pool["curr_n_samples"] == self.training_pool["max_n_samples"]:
            self.training_pool["is_full"] = True
        else:
            self.training_pool["is_full"] = False
    
    @torch.no_grad()
    def dequeue_training_pool(self, data_batch):
    
        # if training pool is full,
        # shuffle the pool, and return randomly selected samples
        bs = data_batch["LR"].shape[0]
        if self.training_pool["is_full"]:
            
            # rand sort
            assert self.training_pool["curr_n_samples"] == self.training_pool["max_n_samples"]
            idx = torch.randperm(self.training_pool["curr_n_samples"])
            self.training_pool["LR"] = self.training_pool["LR"][idx]
            self.training_pool["HR"] = self.training_pool["HR"][idx]
            self.training_pool["LR_deg"] = self.training_pool["LR_deg"][idx]
            self.training_pool["HR_usm"] = self.training_pool["HR_usm"][idx]
            
            # deque (get last n samples)
            ptr = self.training_pool["curr_n_samples"]
            data_batch["LR"] = self.training_pool["LR"][ptr-bs:ptr].clone().to("cuda", non_blocking=True)
            data_batch["HR"] = self.training_pool["HR"][ptr-bs:ptr].clone().to("cuda", non_blocking=True)
            data_batch["LR_deg"] = self.training_pool["LR_deg"][ptr-bs:ptr].clone().to("cuda", non_blocking=True)
            data_batch["HR_usm"] = self.training_pool["HR_usm"][ptr-bs:ptr].clone().to("cuda", non_blocking=True)
            self.training_pool["curr_n_samples"] = ptr - bs  # decrease pointer
            self.training_pool["is_full"] = False
            return data_batch


        # if training pool is not full,
        # simply return the original batch data. (which is at the end of the queue)
        else:
            return data_batch

    # -------------------- degradation --------------------
    
    @torch.no_grad()
    def second_order_degradation(self, data_batch):
        """
        code is based on https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/models/realesrgan_model.py
        """
        gt = data_batch['HR'].clone()
        gt_usm = self.usm_sharpener(gt)
    
        self.kernel1 = data_batch['kernel1'].to(torch.float32)
        self.kernel2 = data_batch['kernel2'].to(torch.float32)
        self.sinc_kernel = data_batch['sinc_kernel'].to(torch.float32)
    
    
        ori_h, ori_w = gt.size()[2:4]
    
        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt_usm, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.deg_opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.deg_opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.deg_opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.deg_opt['gray_noise_prob']
        if np.random.uniform() < self.deg_opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.deg_opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.deg_opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.deg_opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)
    
        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.deg_opt['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.deg_opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.deg_opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.deg_opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.deg_opt['scale'] * scale), int(ori_w / self.deg_opt['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = self.deg_opt['gray_noise_prob2']
        if np.random.uniform() < self.deg_opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.deg_opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.deg_opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
            
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.deg_opt['scale'], ori_w // self.deg_opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.deg_opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.deg_opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.deg_opt['scale'], ori_w // self.deg_opt['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
        
        
        # add generated results
        data_batch["LR_deg"] = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        data_batch["HR_usm"] = gt_usm
        
        return data_batch

    @torch.no_grad()
    def preprocess_batch(self, data_batch):
        data_batch = self.second_order_degradation(data_batch)
        self.enqueue_training_pool(data_batch)
        data_batch = self.dequeue_training_pool(data_batch)
    
        return data_batch

    # -------------------- train functions for multiple inputs --------------------

    def train_1iter(self, data_batch):
    
        self.optimizer.zero_grad(set_to_none=True)
    
        # if "LR2" in data_batch.keys():
    
        self._train_core(data_batch)
    
        self.train_flags.step()  # increase curr_iter, set self.train_flags.STOP
        self.engine_interrupt_per_iter(self)

    def _train_core(self, data_batch, **kwargs):
    
        lr = data_batch["LR"]
        lr_deg = data_batch["LR_deg"]
        hr = data_batch["HR"]
        hr_usm = data_batch["HR_usm"]

        torch.autograd.set_detect_anomaly(True)
        intermediate, sr = self.generator(lr_deg, return_intermediate=True)

        
        if self.train_flags.curr_iter % 100==0:
            save_sr = make_grid(sr[:4, ...].clamp(0,1).clone().detach(), nrow=2).cpu().clamp(0,1)
            save_intermediate = make_grid(intermediate[:4, ...].clamp(0,1).clone().detach(), nrow=2).cpu().clamp(0,1)
            save_lr = make_grid(lr[:4, ...].clamp(0,1).clone().detach(), nrow=2).cpu().clamp(0,1)
            save_lr_deg = make_grid(lr_deg[:4, ...].clamp(0,1).clone().detach(), nrow=2).cpu().clamp(0,1)
            
            
            save_name = self.config.config_name
            os.makedirs(f"/dump/{save_name}/", exist_ok=True)
            to_pil_image(save_sr).save(f"/dump/{save_name}/curr_sr.jpg")
            to_pil_image(save_intermediate).save(f"/dump/{save_name}/curr_intermediate.jpg")
            to_pil_image(save_lr).save(f"/dump/{save_name}/curr_lr.jpg")
            to_pil_image(save_lr_deg).save(f"/dump/{save_name}/curr_lrdeg.jpg")
        
        
        # for full scale output
        pix_loss_full = self.pixel_loss(sr=sr, hr=hr_usm, coef=self.loss_coef["pixel"])
        per_loss_full = self.perceptual_loss(sr=sr, hr=hr_usm, coef=self.loss_coef["percep"])
        loss_full = pix_loss_full + per_loss_full
        
        # for intermediate output
        pix_loss_intermediate = self.pixel_loss(sr=intermediate, hr=lr, coef=self.loss_coef["pixel"])
        per_loss_intermediate = self.perceptual_loss(sr=intermediate, hr=lr, coef=self.loss_coef["percep"])
        loss_intermediate = pix_loss_intermediate + per_loss_intermediate

        self.loss_logger.cache_in("Train/Loss/pix_loss_inter", pix_loss_intermediate.item())
        self.loss_logger.cache_in("Train/Loss/per_loss_inter", per_loss_intermediate.item())
        self.loss_logger.cache_in("Train/Loss/pix_loss_full", pix_loss_full.item())
        self.loss_logger.cache_in("Train/Loss/per_loss_full", per_loss_full.item())
        
        
        
        # adv loss
        # adv_loss_full = self.adv_loss(sr=sr, hr=hr, coef=self.loss_coef["adv"])
        # adv_loss_intermediate = self.adv_loss(sr=intermediate, hr=lr, coef=self.loss_coef["adv"])

        if self.config.intermediate_output_warmup_iter < self.train_flags.curr_iter:
            # D step
            self.adv_loss.optim_D.zero_grad()
            for params in self.adv_loss.discriminator.parameters():
                params.requires_grad = True
    
            adv_loss_d1 = self.loss_coef["adv"] * self.adv_loss._forward_impl_D(sr.detach().clone(), hr)
            adv_loss_d2 = self.loss_coef["adv"] * self.adv_loss._forward_impl_D(intermediate.detach().clone(), lr)
            adv_loss_d = adv_loss_d1 + adv_loss_d2
            adv_loss_d.backward()
            self.adv_loss.optim_D.step()
            self.adv_loss.scheduler.step()
            
            # G step
            for params in self.adv_loss.discriminator.parameters():
                params.requires_grad = False
    
            adv_loss_g1 = self.adv_loss._forward_impl_G(sr, hr)
            adv_loss_g2 = self.adv_loss._forward_impl_G(intermediate, lr)

            loss_full += adv_loss_g1
            loss_intermediate += adv_loss_g2
        
            self.loss_logger.cache_in("Train/Loss/adv_D1", adv_loss_d1.item())
            self.loss_logger.cache_in("Train/Loss/adv_D2", adv_loss_d2.item())
            self.loss_logger.cache_in("Train/Loss/adv_G1", adv_loss_g1.item())
            self.loss_logger.cache_in("Train/Loss/adv_G2", adv_loss_g2.item())

        loss = loss_full + loss_intermediate
        self.loss_logger.cache_out()

        # backward
        if loss.to(torch.float32) > self.train_flags.last_loss * self.config.skip_threshold:
            print(f"skipping this step! too large loss. loss={loss.to(torch.float32).item():.4f}, prev={self.train_flags.last_loss:.4f}")
            print(self.config.skip_threshold)
    
        else:
            loss.backward()
            nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=self.config.grad_clip_value)
        
            self.optimizer.step()
            self.scheduler.step()
            self.train_flags.last_loss = loss.to(torch.float32).item()
            self.wandb_run.log({"lr": self.optimizer.param_groups[0]['lr']},
                               step=self.train_flags.curr_iter * self.config.batch_scale)

    def validate(self):
    
        torch.cuda.empty_cache()
        with torch.no_grad():
            # Initialize the data loader and load the first batch of data
            self.prefetcher_valid.reset()
            data_batch = self.prefetcher_valid.next()
            
            while data_batch is not None:
                lr = data_batch["LR"].to(self.config.device)
                hr = data_batch["HR"].to(self.config.device)
            
                sr, iqa_results = self._validate_core(lr, hr)
                data_batch = self.prefetcher_valid.next()
        
            validation_results = self.iqa_evaluator.cache_out()
            if self.wandb_run is not None:
                self.wandb_run.log(validation_results, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
            
            # Save Image
            gen = self.generator_ema if self.train_flags.curr_iter > self.config.ema_start else self.generator
            gen.eval()
            SAVE_IMG_LR = "/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kcrop/LRbicx4_fp32/0801.png"
            SAVE_IMG_LR = to_tensor(Image.open(SAVE_IMG_LR)).unsqueeze(0).cuda()
            INTER, SAVE_IMG_HR = gen(SAVE_IMG_LR, return_intermediate=True)
            
            SAVE_IMG_HR = to_pil_image(SAVE_IMG_HR[0].cpu().clamp(0,1))
            SAVE_IMG_HR = wandb.Image(SAVE_IMG_HR, caption="sr output full")
            
            INTER = to_pil_image(INTER[0].cpu().clamp(0,1))
            INTER = wandb.Image(INTER, caption="sr output inter")
            
            if self.wandb_run is not None:
                self.wandb_run.log({"SR outputs Full": [SAVE_IMG_HR]}, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
                self.wandb_run.log({"SR outputs Inter": [INTER]}, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
            
        self.generator.train()
        return validation_results


    # # Dont use this for fair comparison
    # def update_gen_ema(self):
    #
    #     if isinstance(self.generator, DataParallelModel) or isinstance(self.generator, DP):
    #         generator_bare = self.generator.module
    #     else:
    #         generator_bare = self.generator
    #
    #
    #     if self.config.gen_ema and self.train_flags.curr_iter == self.config.ema_start:
    #         g_name = self.wandb_run.config.generator
    #         g_name1, g_name2 = g_name.split("@")
    #         self.generator_ema = getattr(importlib.import_module(f"model_arch.{g_name1}_arch"), g_name2)().cuda()
    #
    #         net_g_params = dict(generator_bare.named_parameters())
    #         net_g_ema_params = dict(self.generator_ema.named_parameters())
    #
    #         for k in net_g_ema_params.keys():
    #             net_g_ema_params[k].data.mul_(0).add_(net_g_params[k].data, alpha=1)
    #
    #
    #     if self.config.gen_ema and self.train_flags.curr_iter >= self.config.ema_start:
    #
    #
    #         net_g_params = dict(generator_bare.named_parameters())
    #         net_g_ema_params = dict(self.generator_ema.named_parameters())
    #
    #         for k in net_g_ema_params.keys():
    #             net_g_ema_params[k].data.mul_(self.config.ema_decay).add_(net_g_params[k].data, alpha=1 - self.config.ema_decay)

    @torch.no_grad()
    def debug(self):
        import copy
        # Initialize the data loader and load the first batch of data
        self.generator.train()
        self.prefetcher_train.reset()
        data_batch = self.prefetcher_train.next()
    
        while data_batch is not None and not self.train_flags.STOP:
            # train
            db_original = copy.deepcopy(data_batch)
            data_batch = self.preprocess_batch(data_batch)
            
            e2p = self.ent_to_psnr(data_batch["HR"])
            for i in range(data_batch["LR"].shape[0]):
    
                tmp = data_batch["HR"][i].clone().cpu()
                if e2p[i] > 0.4:
                    os.makedirs("/root/dump/e2p_over0.4", exist_ok=True)
                    to_pil_image(tmp).save(f"/root/dump/e2p_over0.4/" + f"{self.train_flags.curr_iter}_{i}_{e2p[i].item():.3f}.jpg")
                else:
                    os.makedirs("/root/dump/e2p_under0.4", exist_ok=True)
                    to_pil_image(tmp).save(f"/root/dump/e2p_under0.4/" + f"{self.train_flags.curr_iter}_{i}_{e2p[i].item():.3f}.jpg")
            

            print(f"======= iter {self.train_flags.curr_iter} =======")
            print(data_batch["LR"].shape)
            print(data_batch["HR"].shape)
            print("is Equal LR", torch.all(data_batch["LR"]==db_original["LR"]))
            print("is Equal HR", torch.all(data_batch["LR"]==db_original["LR"]))
            print("tp curr n", self.training_pool["curr_n_samples"])
            print("tp max n", self.training_pool["max_n_samples"])
            print(e2p.flatten().to(torch.uint8))
            print()
            
            self.train_flags.step()
            data_batch = self.prefetcher_train.next()
            
            if self.train_flags.curr_iter > 100:
                exit()