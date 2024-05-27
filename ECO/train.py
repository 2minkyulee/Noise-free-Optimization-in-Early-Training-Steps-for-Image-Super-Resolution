import torch.backends.cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm
import argparse
from trainer.engine import Engine
import random
import numpy as np

if __name__ == "__main__":
    
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    #
    # Important!!! keep default values as None, else, it will over-write other configs (including config templates)
    #
    args = argparse.ArgumentParser()
    args.add_argument("--trainer", default=None)
    args.add_argument("--debug", default=False, action="store_true")
    args.add_argument("--loss_string", default=None)
    args.add_argument("--config_template", default=None)
    args.add_argument("--wandb_mode", default=None, help="offline|online|disabled")
    args.add_argument("--wandb_name", default=None)
    args.add_argument("--wandb_subname", default=None, help="sets config_template automatically if set as None")
    args.add_argument("--wandb_id", default=None, help="Used for resuming wandb, ignores wandb_name/subname. If None, do not resume wandb.")
    args = args.parse_args()
    
    
    engine = Engine()
    engine.register_wandb(
        name=args.wandb_name,
        sub_name=args.config_template if args.wandb_subname is None else args.wandb_subname,
        debug=args.debug,
        mode="online" if args.wandb_mode is None else args.wandb_mode,
        wandb_id=args.wandb_id
    )
    engine.register_configs(args)
    engine.register_trainer(engine.config.trainer)  # must call set_trainer at last
    engine.init_save_path()  # todo, should get argument to make run folders like yolo v7
    # engine.engine_interrupt_per_iter = lambda trainer: print(trainer.loss_modules.loss_logger.losses, "\n")
    engine.assert_all_ok()
    engine.run()


