import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_loss(nn.Module):
    def __init__(self, config, loss_logger):
        super(L1_loss, self).__init__()
        self.loss_logger = loss_logger


    def forward(self, sr, hr, coef, **kwargs):
        l1_loss = F.l1_loss(sr, hr)
        self.loss_logger.cache_in("Train/Loss/L1", l1_loss.item())
        return coef * l1_loss

class L1x255_loss(nn.Module):
    def __init__(self, config, loss_logger):
        super(L1x255_loss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, sr, hr, coef, **kwargs):
        
        sr = sr * 255
        hr = hr * 255
        l1_loss = F.l1_loss(sr, hr)
        
        self.loss_logger.cache_in("Train/Loss/L1x255", l1_loss.item())
        
        return coef * l1_loss

class FFT_loss(nn.Module):
    def __init__(self, config, loss_logger):
        super(FFT_loss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, sr, hr, coef, **kwargs):
        
        # only consider mag for now
        sr_fft = torch.fft.rfft2(sr, norm="ortho", dim=(-2, -1))
        hr_fft = torch.fft.rfft2(hr, norm="ortho", dim=(-2, -1))
        
        fft_loss = F.mse_loss(sr_fft.abs(), hr_fft.abs())
        

        self.loss_logger.cache_in("Train/Loss/FFT", fft_loss.item())
        
        ret = coef * fft_loss
        return ret


class FFTx255_loss(nn.Module):
    def __init__(self, config, loss_logger):
        super(FFTx255_loss, self).__init__()
        self.loss_logger = loss_logger
    
    def forward(self, sr, hr, coef, **kwargs):
        
        # only consider mag for now
        sr = sr*255
        hr = hr*255

        sr_fft = torch.fft.rfft2(sr, norm="ortho", dim=(-2, -1))
        hr_fft = torch.fft.rfft2(hr, norm="ortho", dim=(-2, -1))

        fft_loss = F.mse_loss(sr_fft.abs(), hr_fft.abs())

        self.loss_logger.cache_in("Train/Loss/FFTx255", fft_loss.item())
        
        ret = coef * fft_loss
        return ret


class L2_loss(nn.Module):
    def __init__(self, config, loss_logger):
        super(L2_loss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, sr, hr, coef, **kwargs):
        l2_loss = F.mse_loss(sr, hr)
        self.loss_logger.cache_in("Train/Loss/L2", l2_loss.item())
        return coef * l2_loss


class Posterior_loss(nn.Module):
    """
    Simple version
    L_E implementation of "Revisiting `1 Loss in Super-Resolution: A Probabilistic View and Beyond"
    (https://arxiv.org/pdf/2201.10084.pdf)
    """
    def __init__(self, config, loss_logger):
        super(Posterior_loss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, sr, hr, coef, **kwargs):
        
        sigma = torch.abs(sr - hr)
        sigma = torch.where(sigma < sigma.mean(), 0, sigma)
        z = torch.randn_like(sr)
        
        posterior_loss = F.l1_loss(hr, (sr + sigma*z))
        self.loss_logger.cache_in("Train/Loss/Posterior", posterior_loss.item())
        
        return coef * posterior_loss


class Posteriorx255_loss(nn.Module):
    """
    Simple version
    L_E implementation of "Revisiting `1 Loss in Super-Resolution: A Probabilistic View and Beyond"
    (https://arxiv.org/pdf/2201.10084.pdf)
    """
    
    def __init__(self, config, loss_logger):
        super(Posteriorx255_loss, self).__init__()
        self.loss_logger = loss_logger
    
    def forward(self, sr, hr, coef, **kwargs):
        
        sr = sr*255
        hr = hr*255
        
        sigma = torch.abs(sr - hr)
        sigma = torch.where(sigma < sigma.mean(), 0, sigma)
        z = torch.randn_like(sr)
        
        posterior_loss = F.l1_loss(hr, (sr + sigma * z))
        self.loss_logger.cache_in("Train/Loss/Posteriorx255", posterior_loss.item())
        
        return coef * posterior_loss



class AuxPosterior_loss(nn.Module):
    """
    Simple version
    Auxiliary Posterior Loss
    """
    def __init__(self, config, loss_logger):
        super(AuxPosterior_loss, self).__init__()
        self.loss_logger = loss_logger

    def forward(self, sr, hr, coef, **kwargs):
        
        sigma = torch.abs(sr - kwargs["aux"])
        sigma = torch.where(sigma < sigma.mean(), 0, sigma)
        z = torch.randn_like(sr)
        
        posterior_loss = F.l1_loss(hr, (sr + sigma*z))
        self.loss_logger.cache_in("Train/Loss/AuxPosterior", posterior_loss.item())
        
        return coef * posterior_loss


class ModifPosterior_loss(nn.Module):
    """
    Simple version
    Modified version of L_E implementation of "Revisiting `1 Loss in Super-Resolution: A Probabilistic View and Beyond"
    (https://arxiv.org/pdf/2201.10084.pdf)
    """
    
    def __init__(self, config, loss_logger):
        super(ModifPosterior_loss, self).__init__()
        self.loss_logger = loss_logger
    
    def forward(self, sr, hr, coef, **kwargs):
        
        radius = 1
        sigma = 1
        
        random_angle = torch.randn_like(sr)
        random_normal_angle = random_angle / torch.sqrt((random_angle**2).sum())
        
        z_prime = (radius * random_normal_angle)+torch.randn_like(sr)*sigma

        modif_posterior_loss = F.l1_loss(hr, (sr + z_prime))
        self.loss_logger.cache_in("Train/Loss/ModifPosterior", modif_posterior_loss.item())

        return coef * modif_posterior_loss

