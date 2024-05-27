import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossLogger():
    def __init__(self, wandb_run, train_flags):
        self.wandb_run = wandb_run
        self.train_flags = train_flags
        self.losses = {}
    
    def cache_in(self, name, val):
        self.losses[name] = val
        
    def cache_out(self):
        self._cache_out()
        self.clean_cache()
        
    def clean_cache(self):
        self.losses = {}
    
    def _cache_out(self):
        if self.wandb_run is not None:
            self.wandb_run.log(self.losses, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
        
        
        
class Loss(nn.Module):

    def __init__(self, wandb_run, train_flags):

        """
        loss_name = AdversarialLoss_DEFAULT, L1, L2, PerceptualLoss ... etc
        loss_type = adversarial, reconstruction, perceptual ... etc
        """


        super(Loss, self).__init__()
        self.config = wandb_run.config
        self.loss_logger = LossLogger(wandb_run, train_flags)
        self.loss_string = self.config.loss_string
        self.losses_activated = {}

        # @tag [Loss Definitions]
        """
        module_name => loss.xxxx.py
        attr => class name inside loss.module_name.py
        """
        self.losses = {
            "AdversarialLoss_DEFAULT": {
                "module_name": "loss.adversarial_loss",
                "attr": "AdversarialLoss",
                "kwargs": {
                    "discriminator_type":"DEFAULT",
                }
            },
            "AdversarialLoss_HIGHPASS": {
                "module_name": "loss.adversarial_loss",
                "attr": "AdversarialLoss",
                "kwargs": {
                    "discriminator_type":"HIGHPASS",
                }
            },
            "PerceptualLoss": {
                "module_name": "loss.perceptual_loss",
                "attr": "PerceptualLoss",
            },
            "L1": {
                "module_name": "loss.reconstruction_loss",
                "attr": "L1_loss",
            },
            "L1x255": {
                "module_name": "loss.reconstruction_loss",
                "attr": "L1x255_loss",
            },
            "FFT": {
                "module_name": "loss.reconstruction_loss",
                "attr": "FFT_loss",
            },
            "FFTx255": {
                "module_name": "loss.reconstruction_loss",
                "attr": "FFTx255_loss",
            },
            "L2": {
                "module_name": "loss.reconstruction_loss",
                "attr": "L2_loss",
            },
            "Post": {
                "module_name": "loss.reconstruction_loss",
                "attr": "Posterior_loss",
            },
            "Postx255": {
                "module_name": "loss.reconstruction_loss",
                "attr": "Posteriorx255_loss",
            },
            "AuxPost": {
                "module_name": "loss.reconstruction_loss",
                "attr": "AuxPosterior_loss",
            },
            

        }
        
        # assign wandb_run and loss_logger for logging.
        for k, v in self.losses.items():
            if "kwargs" not in self.losses[k].keys():
                self.losses[k]["kwargs"]={}
            self.losses[k]["kwargs"]["config"] = self.config
            self.losses[k]["kwargs"]["loss_logger"] = self.loss_logger

        
        # dynamically define losses based on given string (ex: self.loss_string = "0.1*L1+1*PerceptualLoss+0.1*AdversarialLoss_DEFAULT")
        for coef_loss in self.loss_string.split("+"):
            coef, loss_name = coef_loss.split("*")
            coef = float(coef)

            # dynamic reading from self.losses (non-activated, total list of losses)
            m = self.losses[loss_name]
            module_name = importlib.import_module(m["module_name"])
            loss_fn = getattr(module_name, m["attr"])(**m["kwargs"])


            # self.register(loss_fn)
            self.register_module(loss_name, loss_fn)

            # assign self.losses_activated
            self.losses_activated[loss_name] = {}
            self.losses_activated[loss_name]["coef"] = coef
            self.losses_activated[loss_name]["loss_fn"] = loss_fn  # loss_fn is a instance of nn.Module that returns the loss in the forward() mehtod
            self.losses_activated[loss_name]["loss_value"] = None  # will be evaluated in forward()


    def forward(self, sr, hr, **kwargs):

        # forward through losses
        hr = hr.detach()
        # loss = torch.complex(torch.tensor(0.), torch.tensor(0.)).to('cuda')
        loss = torch.tensor(0., device='cuda')
        for k, v in self.losses_activated.items():
            # loss_name = k
            if v["coef"] != 0.:
                loss += v["loss_fn"](sr, hr, coef=v["coef"], **kwargs)  # DO NOT multiply coef here. It is also used internally for D update.

        
        self.loss_logger.cache_out()

        return loss.real
