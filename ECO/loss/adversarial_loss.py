"""
Modified from BasicSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_arch.custom_modules.frequency import HighPassLayer
import math
from torch.optim import lr_scheduler
from torch.optim import Adam
from utils.train_utils import define_scheduler, define_optimizer
from importlib import import_module
from torch.nn.utils import spectral_norm

class AdversarialLoss(nn.Module):

    def __init__(self,  wandb_run, loss_logger, discriminator_type="DEFAULT", relative=True):
        super(AdversarialLoss, self).__init__()

        # define discriminator
        self.discriminator_type = discriminator_type
        module = import_module(f"loss.adversarial_loss")
        self.discriminator = getattr(module, f"Discriminator_{discriminator_type}")()
        self.discriminator.train()

        # define optimizer, scheduler, loss_fn for internal Discriminator update
        self.optim_D = define_optimizer(self.discriminator, wandb_run.config)
        self.scheduler = define_scheduler(self.optim_D, wandb_run.config)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # misc
        self.register_buffer("_device", torch.tensor(0))  # define "device" to track device of module
        self.loss_logger = loss_logger
        self.wandb_run = wandb_run
        self.relative = relative

    def load_discriminator(self):
        # todo load discriminator
        load_path = self.wandb_run.config.load_path_discriminator[self.discriminator_type]
        self.discriminator.load_state_dict(torch.load(load_path))



    def get_device(self):
        return self._device.device

    def _forward_impl_G(self, sr, hr) -> torch.Tensor:
        """
        forward implementation to obtain grads for G update.
        """
        # Set label.
        # real_label = torch.full([sr.size(0), 1], 1.0, dtype=sr.dtype, device=self.get_device())
        # fake_label = torch.full([sr.size(0), 1], 0.0, dtype=sr.dtype, device=self.get_device())

        # forward through discriminator
        sr_output = self.discriminator(sr)
        hr_output = self.discriminator(hr)

        # calc relative adversarial loss (introduced in ESRGAN)
        real_label = torch.ones_like(hr_output, requires_grad=False)
        fake_label = torch.zeros_like(hr_output, requires_grad=False)
        
        rel_sr = torch.mean(sr_output) if self.relative else 0
        rel_hr = torch.mean(hr_output) if self.relative else 0
        d_loss_hr = self.loss_fn(hr_output - rel_sr, fake_label) * 0.5
        d_loss_sr = self.loss_fn(sr_output - rel_hr, real_label) * 0.5
        adversarial_loss_g = (d_loss_hr + d_loss_sr)

        return adversarial_loss_g

    def _forward_impl_D(self, sr, hr) -> torch.Tensor:
        """
        forward implementation to obtain grads for D update.
        """
        
        # Set label.
        # real_label = torch.full([sr.size(0), 1], 1.0, dtype=sr.dtype, device=self.get_device())
        # fake_label = torch.full([sr.size(0), 1], 0.0, dtype=sr.dtype, device=self.get_device())

        # forward through discriminator
        sr_output = self.discriminator(sr)
        hr_output = self.discriminator(hr)

        # calc relative adversarial loss (introduced in ESRGAN)
        real_label = torch.ones_like(hr_output, requires_grad=False)
        fake_label = torch.zeros_like(hr_output, requires_grad=False)

        rel_sr = torch.mean(sr_output) if self.relative else 0
        rel_hr = torch.mean(hr_output) if self.relative else 0
        d_loss_hr = self.loss_fn(hr_output - rel_sr, real_label) * 0.5
        d_loss_sr = self.loss_fn(sr_output - rel_hr, fake_label) * 0.5
        adversarial_loss_d = (d_loss_hr + d_loss_sr)

        self.loss_logger.cache_in("Train/Loss/adv_D", adversarial_loss_d.item())
        
        with torch.no_grad():
            d_hr_relative_probability = torch.sigmoid_(hr_output.detach() - torch.mean(sr_output.detach())).mean()
            d_sr_relative_probability = torch.sigmoid_(sr_output.detach() - torch.mean(hr_output.detach())).mean()
            self.loss_logger.cache_in("Train/Loss/relative_prob_hr", d_hr_relative_probability.item())
            self.loss_logger.cache_in("Train/Loss/relative_prob_sr", d_sr_relative_probability.item())


        return adversarial_loss_d

    def forward(self, sr, hr, coef, update_D=True, **kwargs):

        # @tag [Forward D] update self.discriminator
        # dont return anything
        
        
        self.optim_D.zero_grad()

        # Set discriminator grad. It was set as false in below (when updating the generator).
        for params in self.discriminator.parameters():
            params.requires_grad = True

        adversarial_loss_d = coef * self._forward_impl_D(sr.detach().clone(), hr)
        adversarial_loss_d.backward()
        
        self.optim_D.step()
        self.scheduler.step()


        # @tag [Forward G]
        # dont update self.discriminator
        
        # D does not require grad when updating G
        for params in self.discriminator.parameters():
            params.requires_grad = False

        adversarial_loss_g = self._forward_impl_G(sr, hr)
        self.loss_logger.cache_in("Train/Loss/adv_G", adversarial_loss_g.item())

        return coef * adversarial_loss_g






# @tag [Define Discriminator_DEFAULT]
class Discriminator_DEFAULT(nn.Module):
    """
    The default discriminator for adversarial loss
    input size with BCHW = B x 3 x 128 128
    """
    def __init__(self):
        super(Discriminator_DEFAULT, self).__init__()
        # misc
        self.register_buffer("_device", torch.tensor(0))  # define "device" to track device of module

        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x):

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


# @tag [Define Discriminator_HIGHPASS]
class Discriminator_HIGHPASS(nn.Module):
    """
    The default discriminator for adversarial loss
    input size with BCHW = B x 3 x 128 128
    """
    def __init__(self):
        super(Discriminator_HIGHPASS, self).__init__()

        # misc
        self.register_buffer("_device", torch.tensor(0))  # define "device" to track device of module
        self.hpf = HighPassLayer(h=128, w=128, sigma=3/math.pi)  # must match input size with below. Default=(128, 128)

        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x):

        gkern, out = self.hpf(x)
        out = self.features(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out



class Discriminator_UNetSN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(Discriminator_UNetSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out