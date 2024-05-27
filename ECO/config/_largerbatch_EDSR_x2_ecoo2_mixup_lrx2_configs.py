from config.base_configs import BaseConfig
import utils.constant
import os



class Config(BaseConfig):

    def __init__(self):
        super(Config, self).__init__()

        # misc
        self.config_name = "largerbatch_EDSR_x2_ecoo2_mixup_lrx2"
        self.mixup_shift = 0
        
        
        # model
        self.generator = "EDSR@EDSR_x2"

        # goto "@tag [Loss Definitions]"
        self.loss_string = "+".join([
            "1*L1x255"
        ])

        # larger_batch
        self.batch_scale = 8

        # dataset & loader
        self.train_scale_factor = 2
        self.train_patch_size = 96
        self.train_batch_size = 16 * self.batch_scale

        # dataset & loader
        self.train_path_imageHR = [
            utils.constant.PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_x2,
        ]
        self.train_path_imageLR = [
            utils.constant.PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_x2,
        ]
        self.train_path_imageHR2 = [
            utils.constant.PREPROCESSED_DATASET_HR_DIV2Ktrain,
        ]
        self.train_path_imageLR2 = [
            utils.constant.PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain,
        ]
        
        self.valid_path_imageHR = [
            utils.constant.DATASET_HR_Set5
        ]
        self.valid_path_imageLR = [
            utils.constant.DATASET_LRBICx2_Set5
        ]

        # optimizer options
        self.optimizer = "Lamb"
        self.optimizer_kwargs = {
            "lr": 1e-4 * self.batch_scale * 2 * 2,
            "betas": (0.9, 0.99),
        }

        # scheduler options
        self.max_iter = 1000000 // self.batch_scale  # == 1000k
        self.scheduler = "WarmupCosineLR"
        self.scheduler_kwargs = {
            "max_iters": self.max_iter,
            "warmup_iters": 0
        }
        
    
        # display
        self.valid_rate = 1000 // self.batch_scale  # 5k
        self.model_save_rate = 100000 // self.batch_scale  # 100k

        # which metric to evaluate "best model"
        self.main_metric = "PSNR"

        