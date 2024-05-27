from utils.misc_utils import load_config_template
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm
import wandb
from trainer.trainer import DefaultTrainer
from utils.debug_utils import null_fn
import importlib
from utils.constant import WANDB_PATH

class Engine():
    
    def __init__(
            self,
            engine_interrupt_per_epoch=null_fn,
            engine_interrupt_per_iter=null_fn,
    ):
        
        self._merged_configs = None
        
        # other assets
        self.wandb_run = None
        self.config = None
        self.trainer = None
        
        self.engine_interrupt_per_epoch = engine_interrupt_per_epoch
        self.engine_interrupt_per_iter = engine_interrupt_per_iter
    
        # flags
        self.WANDB_REGISTERED = False
        self.CONFIG_REGISTERED = False
        self.TRAINER_REGISTERED = False
    

    def register_configs(self, args):

        assert self.WANDB_REGISTERED
        # [configs from arg_parser] and [configs from config_template]
        # merge configs
        self._merged_configs = load_config_template(args.config_template, args)
        
        # update [wandb.configs] with [config_templates and arg_parser configs]
        self.wandb_run.config.update(vars(self._merged_configs), allow_val_change=True)  # config priority 2nd
        # self.wandb_run.config.update(self._args_tmp, allow_val_change=True)  # config priority 1st
        self.config = self.wandb_run.config
        self.CONFIG_REGISTERED = True
        
    def register_wandb(self, name, sub_name, debug, mode, wandb_id):
        
        # set wandb
        wandb_mode = "disabled" if debug else mode
        print(f"WANDB MODE: {wandb_mode}")
        
        # initialize wandb path
        wandb_path = WANDB_PATH # "/mnt/hdd0/mklee/wandb"
        os.makedirs(wandb_path, exist_ok=True)
        
        if wandb_id is None:
            wandb_run = wandb.init(
                project=name,  # this is correct. I didnt confuse project/name vs name/sub_name
                name=sub_name,
                mode=wandb_mode,
                dir=wandb_path
            )
        else:
            wandb_run = wandb.init(
                project=name,
                id=wandb_id,
                resume="must",
                dir=wandb_path
            )
        self.wandb_run = wandb_run
        self.wandb_run.log_code("./trainer/trainer.py")
        self.WANDB_REGISTERED = True

    def register_trainer(self, trainer_type):
        
        # define trainer
        trainer_module = importlib.import_module(f"trainer")
        trainer = getattr(trainer_module, trainer_type)
        self.trainer = trainer(
            wandb_run=self.wandb_run,
            engine_interrupt_per_iter=self.engine_interrupt_per_iter,
            engine_interrupt_per_epoch=self.engine_interrupt_per_epoch
        )
        
        self.TRAINER_REGISTERED = True
        
        
    def init_save_path(self):
        os.makedirs(self.wandb_run.config.save_path_base, exist_ok=True)
        os.makedirs(os.path.dirname(self.wandb_run.config.save_path_generator), exist_ok=True)
        os.makedirs(os.path.dirname(self.wandb_run.config.save_path_optimizer), exist_ok=True)
        os.makedirs(os.path.dirname(self.wandb_run.config.save_path_scheduler), exist_ok=True)
        os.makedirs(os.path.dirname(self.wandb_run.config.save_path_lossmodules), exist_ok=True)

    def assert_all_ok(self):

        assert self.TRAINER_REGISTERED
        assert self.CONFIG_REGISTERED
        assert self.WANDB_REGISTERED
        
        # if resuming wandb, also set load options True
        if self.config.wandb_id is not None:
            assert self.config.load_trainflags
            assert self.config.load_generator
            assert self.config.load_lossmodules
            assert self.config.load_optimizer
            assert self.config.load_scheduler

    def run(self):
        self.trainer.load_modules()
        self.trainer.run()

    def prepare_dataset(self):
        pass
    
