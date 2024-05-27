import utils.constant
import datetime
import os
import torch.optim
import wandb
from utils import debug_utils
from utils.constant import BASE_PATH

# basic configurations. will be over written if custom configs are given.

class BaseConfig:
    
    def __init__(self):

        # server
        try:
            with open("/mnt/hdd0/mklee/server_name.txt") as file:
                self.server_ip = file.read().rstrip()
        except Exception as e:
            print(e)
            print("server name not found")
            self.server_ip = "server name not found"
        
        
        # misc
        self.wandb_id = None
        self.trainer = "DefaultTrainer"
        self.verbose = 999
        self.update_flag = False
        self.device = "cuda"
        self.config_name = None
        self.config_name_auto_add_time = True
        self.shuffle_train_data = True
        self.alpha_sample = False
        self.batch_scale = 1   # must manually adjust iter, logging step, learning rate ... etc
        self.mixup_shift = False
        
        # training heuristics to avoid nan (especially in RCAN)
        self.skip_threshold = 5
        self.grad_clip_value = 0.1

        # network ema
        self.gen_ema = False
        self.ema_start = 5000
        self.ema_decay = 0.999


        # resume (load) options
        self.resume_iter = 0   # for resuming within a single training process --> curr_iter
        self.load_iter = 0     # load checkpoints. Generally, resume_iter==load_iter. (resume_iter != load_iter when using as pretrained model)
        self.load_trainflags = False
        self.load_generator = False
        self.load_lossmodules = False
        self.load_optimizer = False
        self.load_scheduler = False

        
        self.load_path_base = f"{BASE_PATH}/mklee_SR_framework/saved_results/@[config_name]"
        self.load_path_trainflags =  os.path.join(self.load_path_base, "iter@[load_iter]_trainflags.pth")
        self.load_path_generator =   os.path.join(self.load_path_base, "iter@[load_iter]_generator.pth")
        self.load_path_optimizer =   os.path.join(self.load_path_base, "iter@[load_iter]_optimizer.pth")
        self.load_path_scheduler =   os.path.join(self.load_path_base, "iter@[load_iter]_scheduler.pth")
        self.load_path_lossmodules = os.path.join(self.load_path_base, "iter@[load_iter]_lossmodule.pth")

        self.save_path_base = f"{BASE_PATH}/mklee_SR_framework/training_results/@[config_name]"
        self.save_path_trainflags =  os.path.join(self.save_path_base, "iter@[save_iter]_trainflags.pth")
        self.save_path_generator =   os.path.join(self.save_path_base, "iter@[save_iter]_generator.pth")
        self.save_path_generator_ema =   os.path.join(self.save_path_base, "iter@[save_iter]_generator_ema.pth")
        self.save_path_optimizer =   os.path.join(self.save_path_base, "iter@[save_iter]_optimizer.pth")
        self.save_path_scheduler =   os.path.join(self.save_path_base, "iter@[save_iter]_scheduler.pth")
        self.save_path_lossmodules = os.path.join(self.save_path_base, "iter@[save_iter]_lossmodule.pth")
        
        
        # dataset & loader
        self.train_aug = True
        self.train_path_imageHR = ""
        self.train_path_imageLR = ""
        self.train_path_imageHR2 = ""
        self.train_path_imageLR2 = ""
        
        self.train_path_imageAUX = ""
        
        self.train_scale_factor = 4
        self.train_patch_size = 192
        self.train_batch_size = 16
        
        self.valid_path_imageHR = ""
        self.valid_path_imageLR = ""
        self.valid_path_imageHR2 = ""  # probably will not use it.
        self.valid_path_imageLR2 = ""  # probably will not use it.
        
        self.valid_path_imageAUX = ""
        self.valid_scale_factor = 4
        self.valid_patch_size = 0  # dont crop if 0
        self.valid_batch_size = 1
        

        # self.test_path_imageHR = ""
        # self.test_path_imageLR = ""
        # self.test_scale_factor = 4
        # self.test_patch_size = 0  # dont crop if 0
        # self.test_batch_size = 1

        # dataloader misc
        self.pin_memory = True
        self.num_workers = 8

        
        # valid_rate(= validation rate = display rate), save rate (in terms of iter)
        self.wandb_validation_image_visualization = False
        self.valid_rate = 500
        self.model_save_rate = 100000
        
        # which metric to evaluate "best model"
        self.METRICS = ["PSNR", "SSIM"]  # list of metrics to evaluate on
        self.main_metric = "TODO"
        self.eval_crop_border = -1
        self.eval_y_channel = True

    # todo: Doesnt work when called by student classes
    @classmethod
    def get_class_name(cls):
        return cls.__module__

    # @tag [Update]
    def update(self):
        """
        Attribute like self.load_path (or other attributes containing the identifier @ to be replaced, such as @config_name)
        should defined after self.config_name.
        In subclasses as config.train_ESRGAN_configs.Config,
        attributes like self.load_path should be re-initialized
        (i.e. updated) to their corresponding config_name.
        """

        self.valid_scale_factor = self.train_scale_factor



        if self.config_name_auto_add_time:
            now = datetime.datetime.now()
            now = now.strftime("%Y%m%d_%H%M%S")                # YYYYMMDD_HHMMSS format
            self.config_name = self.config_name + f"_{now}"  # add date postfix to prevent conflicting config_names

        self.update_flag = True
        
        for k, v in vars(self).items():  # for all attributes
            if type(v)==str and "@" in v:  # probably paths
                for k_, v_ in vars(self).items():  # for all attributes to replace (starting with @)
                    _this_will_be_replaced_as = f"@[{k_}]"
                    _this = str(v_)
                    vars(self)[k] = vars(self)[k].replace(_this_will_be_replaced_as, _this)

                    

    def assert_all_ok(self):
        assert self.update_flag, "config_template not updated. Must explicitly call config_template.update()"
        assert self.config_name is not None, f"project name not defined in {self.__class__}"


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
