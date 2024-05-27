from config.base_configs import BaseConfig
import utils.constant



class Config(BaseConfig):

    def __init__(self):
        super(Config, self).__init__()

        # misc
        self.config_name = "largerbatch_EDSR_S_x2_default"
        self.num_workers = 32
        
        # model
        self.generator = "EDSR@EDSR_S_x2"

        # goto "@tag [Loss Definitions]"
        self.loss_string = "+".join([
            "1*L1x255"
        ])

        # larger_batch
        self.batch_scale = 8
        self.num_workers = 32

        # dataset & loader
        self.train_scale_factor = 2
        self.train_patch_size = 96
        self.train_batch_size = 16 * self.batch_scale

        # dataset & loader
        self.train_path_imageHR = [
            utils.constant.PREPROCESSED_DATASET_HR_DIV2Ktrain,
        ]
        self.train_path_imageLR = [
            utils.constant.PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain,
        ]
        self.valid_path_imageHR = [
            utils.constant.DATASET_HR_DIV2Kmini
        ]
        self.valid_path_imageLR = [
            utils.constant.DATASET_LRBICx2_DIV2Kmini
        ]

        # optimizer options
        self.optimizer = "Lamb"
        self.optimizer_kwargs = {
            "lr": 1e-4 * self.batch_scale,
            "betas": (0.9, 0.99),
        }

        # scheduler options
        self.max_iter = 1000000 // self.batch_scale  # == 1000k
        self.scheduler = "StepLR"
        self.scheduler_kwargs = {
            "step_size": 200000 // self.batch_scale,
            "gamma": 0.5
        }
        
    
        # display
        self.valid_rate = 1000 // self.batch_scale  # 5k
        self.model_save_rate = 100000 // self.batch_scale  # 100k

        # which metric to evaluate "best model"
        self.main_metric = "PSNR"

