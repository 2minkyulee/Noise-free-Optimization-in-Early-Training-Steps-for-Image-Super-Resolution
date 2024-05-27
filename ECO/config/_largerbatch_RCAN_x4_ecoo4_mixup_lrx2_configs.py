from config.base_configs import BaseConfig
import utils.constant
import os


class Config(BaseConfig):

    def __init__(self):
        super(Config, self).__init__()

        # misc
        self.config_name = "largerbatch_RCAN_x4_ecoo4_mixup_lrx2"
        self.mixup_shift = 0
        self.num_workers = 32
        
        # model
        self.generator = "RCAN@RCAN_x4"

        # goto "@tag [Loss Definitions]"
        self.loss_string = "+".join([
            "1*L1x255"
        ])

        # larger_batch
        self.batch_scale = 8

        # dataset & loader
        self.train_scale_factor = 4
        self.train_patch_size = 192
        self.train_batch_size = 16 * self.batch_scale

        # dataset & loader
        self.train_path_imageHR = [
            utils.constant.PREPROCESSED_DATASET_HR_DIV2Ktrain_RCAN_x4,
        ]
        self.train_path_imageLR = [
            utils.constant.PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_RCAN_x4,
        ]
        self.train_path_imageHR2 = [
            utils.constant.PREPROCESSED_DATASET_HR_DIV2Ktrain,
        ]
        self.train_path_imageLR2 = [
            utils.constant.PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain,
        ]
        
        self.valid_path_imageHR = [
            utils.constant.DATASET_HR_Set5
        ]
        self.valid_path_imageLR = [
            utils.constant.DATASET_LRBICx4_Set5
        ]

        # pretrained
        self.load_generator = True
        self.load_path_base = f"/mnt/hdd0/mklee/mklee_SR_framework/saved_parameters/RCAN"
        self.load_path_generator = os.path.join(self.load_path_base, "rcan_x2_ECO2_batchx8_cosine_best.pth")


        # optimizer options
        self.optimizer = "Lamb"
        self.optimizer_kwargs = {
            "lr": 1e-4 * self.batch_scale * 2,
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
        self.valid_rate = 2000 // self.batch_scale  # 5k
        self.model_save_rate = 20000 // self.batch_scale  # 100k

        # which metric to evaluate "best model"
        self.main_metric = "PSNR"

