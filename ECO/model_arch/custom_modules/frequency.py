import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as transforms
import math
import scipy.signal as signal


class HighPassLayer(nn.Module):

    def __init__(self, h=128, w=128, sigma=3/math.pi):
        super(HighPassLayer, self).__init__()
        self.sigma = sigma

        self.h = h
        self.w = w
        self.register_buffer("gkern", self._guassian_kernel_freq(h, w, self.sigma))


    def _guassian_kernel_freq(self, h, w, sigma_img):
        # sigma_freq = 1 / (2*pi*sigma_img)

        def _sigma(size):
            scale_factor = 4
            sigma_freq = 1 / (math.pi * sigma_img)

            return (sigma_freq * size) / scale_factor

        gkern1d_h = torch.from_numpy(signal.gaussian(h, std=_sigma(h))).type(torch.float32)
        gkern1d_w = torch.from_numpy(signal.gaussian(w, std=_sigma(w))).type(torch.float32)
        gkern2d = torch.outer(gkern1d_h, gkern1d_w)
        gkern2d = (1-gkern2d).clamp(0, 1)  # highpass == complementary part of lowpass

        gkern2d = torch.where(gkern2d>0.5, 1., 0.)  # threshold


        return gkern2d


    def forward(self, x):

        b, c, h, w = x.shape

        # dynamic kernel assignment when input changed. Use with care.
        if h != self.h or w!=self.w:
            self.register_buffer("gkern", self._guassian_kernel_freq(h, w, self.sigma))
            self.h = h  # update self.h and self.w with new input size
            self.w = w

        # highpass filtering
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(self.gkern) * x
        x = torch.fft.ifft2(x)
        x = x.real.clamp(0, 1)

        return self.gkern, x