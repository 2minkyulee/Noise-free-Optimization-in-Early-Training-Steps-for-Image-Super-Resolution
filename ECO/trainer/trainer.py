import torch.nn as nn
import os
import random
import wandb
import torch
from utils.debug_utils import null_fn
from utils.train_utils import define_loader, define_optimizer, define_scheduler, CUDAPrefetcher, TrainFlags
from utils.misc_utils import is_better, metric_init, printl
from utils.iqa_evaluate_utils import IQA_evaluator
from utils.img_proc_utils import modcrop
from loss import Loss
import importlib
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import warnings
from torch.nn import DataParallel as DP
warnings.filterwarnings("ignore")


class DefaultTrainer:
    
    def __init__(self,
                 wandb_run,
                 engine_interrupt_per_iter=null_fn,
                 engine_interrupt_per_epoch=null_fn,
                 ):
        
        # engine interrupt
        self.engine_interrupt_per_iter = engine_interrupt_per_iter
        self.engine_interrupt_per_epoch = engine_interrupt_per_epoch
        
        # misc
        self.wandb_run = wandb_run              # wandb run instance
        self.config = self.wandb_run.config     # [config/base_configs.py + config/xxx_xxx_configs.py + arg_parser configs] --(merge)--> wandb.config
        self.device = self.config.device        # device
        self.verbose = self.config.verbose      # vprint level.
        self.METRICS = self.config.METRICS      # metrics to evaluate
        
        # logging
        os.makedirs(self.config.save_path_base)
        self.logfile = open(os.path.join(self.config.save_path_base, "log.txt"), "a")
        
        # train assets
        self.vprint(f"Initializing Train Prefetcher")
        self.prefetcher_train = CUDAPrefetcher(define_loader(mode="train", config=self.config), self.device)
        self.vprint(f"Initializing Valid Prefetcher")
        self.prefetcher_valid = CUDAPrefetcher(define_loader(mode="valid", config=self.config), self.device)
        self.vprint(f"Initializing Loss/TrainFlags/IQA_evaluator Prefetcher")
        self.train_flags = TrainFlags(0, self.config.max_iter, self.config.METRICS)
        self.loss_modules = Loss(self.wandb_run, self.train_flags).to(self.device)
        self.iqa_evaluator = IQA_evaluator(
            metrics=self.METRICS,
            crop_border=self.config.eval_crop_border if self.config.eval_crop_border>0 else self.config.valid_scale_factor,
            test_y_channel=self.config.eval_y_channel,
            do_quantize=True
        )
        
        # Generator (Discriminator inside self.loss_modules)
        self.vprint("Initializing (Generator, Optimizer, Scheduler)")
        g_name = wandb_run.config.generator
        g_name1, g_name2 = g_name.split("@")
        self.generator = getattr(importlib.import_module(f"model_arch.{g_name1}_arch"), g_name2)()
        self.generator_ema = getattr(importlib.import_module(f"model_arch.{g_name1}_arch"), g_name2)()
        
        # Optimizer & Scheduler
        self.optimizer = define_optimizer(self.generator, wandb_run.config)
        self.scheduler = define_scheduler(self.optimizer, wandb_run.config)
        
        # load
        # todo
        self.vprint("\n".join([
            f"Trainer      : {self.config.trainer}",
            f"Loss      : {self.config.loss_string}",
            f"Generator : {self.config.generator}",
            f"Optimizer : {self.config.optimizer}, {self.config.optimizer_kwargs}",
            f"Scheduler : {self.config.scheduler}, {self.config.scheduler_kwargs}",
            f"Gen_EMA : {self.config.gen_ema}",
            f"Template : {self.config.config_template}",
            f"HR_path : {self.config.train_path_imageHR}",
            f"LR_path : {self.config.train_path_imageLR}",
            f"NumWorkers : {self.config.num_workers}",
        ]), level=2)

    def post_process(self):
        pass

    def update_gen_ema(self):
    
        if isinstance(self.generator, DP):
            generator_bare = self.generator.module
        else:
            generator_bare = self.generator
    
            
        if self.config.gen_ema and self.train_flags.curr_iter >= self.config.ema_start:
    
            # move to same device
            dev_gen_bare = next(self.generator.parameters()).device
            dev_gen_ema = next(self.generator_ema.parameters()).device
            if dev_gen_bare != dev_gen_ema:
                self.generator_ema = self.generator_ema.to(dev_gen_bare)
    
            # copy weights if first ema step
            alpha = 1 if self.train_flags.curr_iter == self.config.ema_start else self.config.ema_decay
    
            # ema implementation
            net_g_params = dict(generator_bare.named_parameters())
            net_g_ema_params = dict(self.generator_ema.named_parameters())
            for k in net_g_ema_params.keys():
                net_g_ema_params[k].data.mul_(alpha).add_(net_g_params[k].data, alpha=1 - alpha)
        



    def vprint(self, *args, **kwargs):
        # verbose print
        
        verbose_thresh = kwargs.pop("level") if "level" in kwargs.keys() else 1
        print(*args, **kwargs, file=self.logfile)

        if self.verbose >= verbose_thresh:  # only print out if appropriate verbose-level
            print(*args, **kwargs)

            
    
    def save_modules(self, repl_a, repl_b):
        # todo save wandb
        torch.save(self.train_flags, self.config.save_path_trainflags.replace(repl_a, repl_b))
        torch.save(self.loss_modules.state_dict(), self.config.save_path_lossmodules.replace(repl_a, repl_b))
        torch.save(self.optimizer.state_dict(), self.config.save_path_optimizer.replace(repl_a, repl_b))
        torch.save(self.scheduler.state_dict(), self.config.save_path_scheduler.replace(repl_a, repl_b))
        
        
        # save generator (maybe DataParallel) and generator_ema
        if isinstance(self.generator, DP):
            generator_bare = self.generator.module
        else:
            generator_bare = self.generator
        torch.save(generator_bare.state_dict(), self.config.save_path_generator.replace(repl_a, repl_b))
        
        if self.config.gen_ema and self.train_flags.curr_iter > self.config.ema_start:
            torch.save(self.generator_ema.state_dict(), self.config.save_path_generator_ema.replace(repl_a, repl_b))
    
    
    
    def load_modules(self):
        
        # todo load wandb
        if self.config.load_trainflags:
            self.train_flags = torch.load(self.config.load_path_trainflags)
            if self.train_flags.max_iter < self.config.max_iter:
                self.train_flags.max_iter = self.config.max_iter
                self.train_flags.STOP = False
            
            
        
        if self.config.load_generator:
            print("loading generator")
            ckpt = torch.load(self.config.load_path_generator)
            if "params_ema" in ckpt.keys():
                ckpt = ckpt["params_ema"]
            self.generator.load_state_dict_wrapper(ckpt)
    
            if self.config.gen_ema:
                print("loading generator ema")
                ckpt = torch.load(self.config.load_path_generator_ema)
                if "params_ema" in ckpt.keys():
                    ckpt = ckpt["params_ema"]
                self.generator_ema.load_state_dict_wrapper(ckpt)
    
    
        if self.config.load_lossmodules:
            ckpt = torch.load(self.config.load_path_lossmodules)
            self.loss_modules.load_state_dict(ckpt)
        
        if self.config.load_optimizer:
            ckpt = torch.load(self.config.load_path_optimizer)
            self.optimizer.load_state_dict(ckpt)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
        if self.config.load_scheduler:
            ckpt = torch.load(self.config.load_path_scheduler)
            self.scheduler.load_state_dict(ckpt)
            

            
    
    
    
    def update_best(self, validation_results):
        
        # if main metric not included in the loaded train_flag, add it.
        main_metric = self.config.main_metric
        if main_metric not in self.train_flags.best_metric.keys():
            self.train_flags.best_metric[main_metric] = metric_init(main_metric)
        
        # Save best checkpoint
        if is_better(this_is=validation_results,
                     better_than=self.train_flags.best_metric,
                     metric=main_metric):
            self.vprint(f"saving best model at iter {self.train_flags.curr_iter} with {validation_results}")
            self.train_flags.best_metric = validation_results
            # self.save_modules("iter@[save_iter]", "best_iter" + str(self.train_flags.curr_iter))
            self.save_modules("iter@[save_iter]", "best")
            
    def _update_mixup_alpha(self):
        
        # simple approximation.
        # may use linear scheduling, since we did not observe any significant difference based on the scheduling.
        
        def _sigmoid(x, x_axis_shift=-2):
            
            x = x-x_axis_shift
            x = float(1 / (1 + np.exp(-1*x)))
            x = 1.2 * (x - 0.5)
            x = max(x, -0.5)
            x = min(x, 0.5)
            x = x + 0.5
            
            return x
        
        if hasattr(self.config, "mixup_shift"):
            a = self.train_flags.curr_iter / self.config.max_iter
            a = 10 * (a - 0.5)    # convert 0~1 to -5~5
            a = _sigmoid(a, self.config.mixup_shift)  # convert -5~5 to 0~1
            a = 1 - a             # invert 0~1 to 1~0
        
            self.prefetcher_train.original_dataloader.dataset.mixup_alpha = a
            
            if self.wandb_run is not None:
                self.wandb_run.log({"mixup_alpha" : a}, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
    
    
    def trainval_1epoch(self):
        
        # Initialize the data loader and load the first batch of data
        self.generator.train()
        self.prefetcher_train.reset()
        data_batch = self.prefetcher_train.next()
        
        while data_batch is not None and not self.train_flags.STOP:

            # save checkpoint
            if self.train_flags.curr_iter % self.config.model_save_rate == 0 and self.train_flags.curr_iter != 0:
                self.save_modules("@[save_iter]", str(self.train_flags.curr_iter))
            
            # train
            self._update_mixup_alpha()
            data_batch = self.preprocess_batch(data_batch)
            self.train_1iter(data_batch)
            self.update_gen_ema()
            data_batch = self.prefetcher_train.next()
            
            # validate and save if best
            if self.train_flags.curr_iter % self.config.valid_rate == 0:
                validation_results = self.validate()
                self.update_best(validation_results)

                print(f"valid {validation_results} at iter {self.train_flags.curr_iter}, lr={self.optimizer.param_groups[0]['lr']}")
                
        
        # external event from engine
        self.engine_interrupt_per_epoch(self)
        
        if self.train_flags.STOP:  # save last model
            self.save_modules("@[save_iter]", str(self.train_flags.curr_iter))
    

    def preprocess_batch(self, data_batch):
        return data_batch

    def train_1iter(self, data_batch):
        
        
        lr = data_batch["LR"] #.to(self.config.device)
        hr = data_batch["HR"] #.to(self.config.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # if "LR2" in data_batch.keys():
        
        self._train_core(lr, hr)

        self.train_flags.step()  # increase curr_iter, set self.train_flags.STOP
        self.engine_interrupt_per_iter(self)

    
    def _train_core(self, lr, hr, **kwargs):

        sr = self.generator(lr)
        loss = self.loss_modules(sr, hr, **kwargs)
        
        if loss.to(torch.float32) > self.train_flags.last_loss * self.config.skip_threshold:
            print(f"skipping this step! too large loss. loss={loss.to(torch.float32).item():.4f}")
            
            
        else:
            loss.backward()
            nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=self.config.grad_clip_value)
            
            self.optimizer.step()
            self.scheduler.step()
            self.train_flags.last_loss = loss.to(torch.float32).item()
            if self.wandb_run is not None:
                self.wandb_run.log({"lr" : self.optimizer.param_groups[0]['lr']},
                                   step=self.train_flags.curr_iter * self.config.batch_scale)
        
    
    def validate(self):
        
        torch.cuda.empty_cache()  # remove training batches before validation
        with torch.no_grad():
            # Initialize the data loader and load the first batch of data
            self.prefetcher_valid.reset()
            data_batch = self.prefetcher_valid.next()
            
            while data_batch is not None:
                lr = data_batch["LR"].to(self.config.device)
                hr = data_batch["HR"].to(self.config.device)
                
                sr, iqa_results = self._validate_core(lr, hr)
                
                data_batch = self.prefetcher_valid.next()
            
            validation_results = self.iqa_evaluator.cache_out()
            if self.wandb_run is not None:
                self.wandb_run.log(validation_results, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
            
            # select only one image to display
            if self.config.wandb_validation_image_visualization:
                if self.wandb_run is not None:
                    b,c,h,w = sr.shape
                    sr = to_pil_image(sr[random.randint(0, b - 1), ...])
                    sr = wandb.Image(sr, caption="sr output")
                    self.wandb_run.log({"SR outputs": [sr]}, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
        
        self.generator.train()
        return validation_results
    
    def _validate_core(self, lr, hr):
        
        gen = self.generator_ema if self.config.gen_ema and (self.train_flags.curr_iter>self.config.ema_start) else self.generator
        if isinstance(gen, DP):
            gen = gen.module
            
        gen.eval()
        sr = gen(lr)
        
        sr = modcrop(sr, self.config.valid_scale_factor)
        hr = modcrop(hr, self.config.valid_scale_factor)
        iqa_results = self.iqa_evaluator(sr, hr, metrics=self.config.METRICS)
        
        return sr, iqa_results
    
    def run(self):
        
        
        if torch.cuda.device_count()>1:
            print("Using DataParallel")
            self.generator = self.generator.cpu()
            self.generator = DP(self.generator)
            
        
        self.generator = self.generator.cuda()
        if self.config.gen_ema:
            self.generator_ema = self.generator_ema.cuda()
        
        
        while not self.train_flags.STOP:
            self.train_flags.tic()
            
            self.trainval_1epoch()
            
            self.train_flags.toc()
            self.train_flags.print_times()
        
        self.post_process()


class DefaultTrainer_GPUefficient():
    
    def __init__(self,
                 wandb_run,
                 engine_interrupt_per_iter=null_fn,
                 engine_interrupt_per_epoch=null_fn,
                 ):
        
        
        #
        print("Initializing Trainer:", self.__class__.__name__)
        
        
        # engine interrupt
        self.engine_interrupt_per_iter = engine_interrupt_per_iter
        self.engine_interrupt_per_epoch = engine_interrupt_per_epoch
        
        # misc
        self.wandb_run = wandb_run  # wandb run instance
        self.config = self.wandb_run.config  # [config/base_configs.py + config/xxx_xxx_configs.py + arg_parser configs] --(merge)--> wandb.config
        self.device = self.config.device  # device
        self.verbose = self.config.verbose  # vprint level.
        self.METRICS = self.config.METRICS  # metrics to evaluate
        
        # logging
        os.makedirs(self.config.save_path_base)
        self.logfile = open(os.path.join(self.config.save_path_base, "log.txt"), "a")
        
        # train assets
        self.vprint(f"Initializing Train Loader")
        self.train_loader = define_loader(mode="train", config=self.config)
        self.vprint(f"Initializing Valid Loader")
        self.valid_loader = define_loader(mode="valid", config=self.config)
        self.vprint(f"Initializing Loss/TrainFlags/IQA_evaluator Prefetcher")
        self.train_flags = TrainFlags(0, self.config.max_iter, self.config.METRICS)
        self.loss_modules = Loss(self.wandb_run, self.train_flags).to(self.device)
        self.iqa_evaluator = IQA_evaluator(
            metrics=self.METRICS,
            crop_border=self.config.eval_crop_border if self.config.eval_crop_border > 0 else self.config.valid_scale_factor,
            test_y_channel=self.config.eval_y_channel,
            do_quantize=True
        )
        
        # Generator (Discriminator inside self.loss_modules)
        self.vprint("Initializing (Generator, Optimizer, Scheduler)")
        g_name = wandb_run.config.generator
        g_name1, g_name2 = g_name.split("@")
        self.generator = getattr(importlib.import_module(f"model_arch.{g_name1}_arch"), g_name2)()
        
        # Optimizer & Scheduler
        self.optimizer = define_optimizer(self.generator, wandb_run.config)
        self.scheduler = define_scheduler(self.optimizer, wandb_run.config)
        
        # load
        self.vprint("\n".join([
            f"Trainer      : {self.config.trainer}",
            f"Loss      : {self.config.loss_string}",
            f"Generator : {self.config.generator}",
            f"Optimizer : {self.config.optimizer}, {self.config.optimizer_kwargs}",
            f"Scheduler : {self.config.scheduler}, {self.config.scheduler_kwargs}",
            f"Template : {self.config.config_template}",
            f"HR_path : {self.config.train_path_imageHR}",
            f"LR_path : {self.config.train_path_imageLR}",
            f"NumWorkers : {self.config.num_workers}",
        ]), level=2)
    
    def post_process(self):
        pass
    
    def update_gen_ema(self):
        pass
    
    def vprint(self, *args, **kwargs):
        # verbose print
        
        verbose_thresh = kwargs.pop("level") if "level" in kwargs.keys() else 1
        print(*args, **kwargs, file=self.logfile)
        
        if self.verbose >= verbose_thresh:  # only print out if appropriate verbose-level
            print(*args, **kwargs)
    
    def save_modules(self, repl_a, repl_b):
        # todo save wandb
        torch.save(self.train_flags, self.config.save_path_trainflags.replace(repl_a, repl_b))
        torch.save(self.loss_modules.state_dict(), self.config.save_path_lossmodules.replace(repl_a, repl_b))
        torch.save(self.optimizer.state_dict(), self.config.save_path_optimizer.replace(repl_a, repl_b))
        torch.save(self.scheduler.state_dict(), self.config.save_path_scheduler.replace(repl_a, repl_b))
        
        # save generator (maybe DataParallel) and generator_ema
        if isinstance(self.generator, DP):
            generator_bare = self.generator.module
        else:
            generator_bare = self.generator
        torch.save(generator_bare.state_dict(), self.config.save_path_generator.replace(repl_a, repl_b))
        
        
    def load_modules(self):
        
        # todo load wandb
        if self.config.load_trainflags:
            self.train_flags = torch.load(self.config.load_path_trainflags)
            if self.train_flags.max_iter < self.config.max_iter:
                self.train_flags.max_iter = self.config.max_iter
                self.train_flags.STOP = False
        
        if self.config.load_generator:
            print("loading generator")
            ckpt = torch.load(self.config.load_path_generator)
            if "params_ema" in ckpt.keys():
                ckpt = ckpt["params_ema"]
            self.generator.load_state_dict_wrapper(ckpt)
        
        if self.config.load_lossmodules:
            ckpt = torch.load(self.config.load_path_lossmodules)
            self.loss_modules.load_state_dict(ckpt)
        
        if self.config.load_optimizer:
            ckpt = torch.load(self.config.load_path_optimizer)
            self.optimizer.load_state_dict(ckpt)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
        if self.config.load_scheduler:
            ckpt = torch.load(self.config.load_path_scheduler)
            self.scheduler.load_state_dict(ckpt)
    
    def update_best(self, validation_results):
        
        # if main metric not included in the loaded train_flag, add it.
        main_metric = self.config.main_metric
        if main_metric not in self.train_flags.best_metric.keys():
            self.train_flags.best_metric[main_metric] = metric_init(main_metric)
        
        # Save best checkpoint
        if is_better(this_is=validation_results,
                     better_than=self.train_flags.best_metric,
                     metric=main_metric):
            self.vprint(f"saving best model at iter {self.train_flags.curr_iter} with {validation_results}")
            self.train_flags.best_metric = validation_results
            # self.save_modules("iter@[save_iter]", "best_iter" + str(self.train_flags.curr_iter))
            self.save_modules("iter@[save_iter]", "best")
    
    def _update_mixup_alpha(self):
        pass
    
    def trainval_1epoch(self):
        
        # Initialize the data loader and load the first batch of data
        self.generator.train()
        
        for data_batch in self.train_loader:
            
            if self.train_flags.STOP:
                break
            
            # save checkpoint
            if self.train_flags.curr_iter % self.config.model_save_rate == 0 and self.train_flags.curr_iter != 0:
                self.save_modules("@[save_iter]", str(self.train_flags.curr_iter))
            
            # train
            self._update_mixup_alpha()
            data_batch = self.preprocess_batch(data_batch)
            self.train_1iter(data_batch)
            
            # validate and save if best
            if self.train_flags.curr_iter % self.config.valid_rate == 0:
                validation_results = self.validate()
                self.update_best(validation_results)
                
                print(f"valid {validation_results} at iter {self.train_flags.curr_iter}, lr={self.optimizer.param_groups[0]['lr']}")
        
        # external event from engine
        self.engine_interrupt_per_epoch(self)
        
        if self.train_flags.STOP:  # save last model
            self.save_modules("@[save_iter]", str(self.train_flags.curr_iter))
    
    def preprocess_batch(self, data_batch):
        data_batch["LR"] = data_batch["LR"].cuda()
        data_batch["HR"] = data_batch["HR"].cuda()
        return data_batch
    
    def train_1iter(self, data_batch):
        
        lr = data_batch["LR"]  # .to(self.config.device)
        hr = data_batch["HR"]  # .to(self.config.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # if "LR2" in data_batch.keys():
        
        self._train_core(lr, hr)
        
        self.train_flags.step()  # increase curr_iter, set self.train_flags.STOP
        self.engine_interrupt_per_iter(self)
    
    def _train_core(self, lr, hr, **kwargs):
        
        sr = self.generator(lr)
        loss = self.loss_modules(sr, hr, **kwargs)
        
        if loss.to(torch.float32) > self.train_flags.last_loss * self.config.skip_threshold:
            print(f"skipping this step! too large loss. loss={loss.to(torch.float32).item():.4f}")
        
        
        else:
            loss.backward()
            nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=self.config.grad_clip_value)
            
            self.optimizer.step()
            self.scheduler.step()
            self.train_flags.last_loss = loss.to(torch.float32).item()
            if self.wandb_run is not None:
                self.wandb_run.log({"lr": self.optimizer.param_groups[0]['lr']},
                                   step=self.train_flags.curr_iter * self.config.batch_scale)
    
    def validate(self):
        
        torch.cuda.empty_cache()  # remove training batches before validation
        with torch.no_grad():
            for data_batch in self.valid_loader:
                # data_batch = data_batch.cuda()
                lr = data_batch["LR"].cuda()
                hr = data_batch["HR"].cuda()
                sr, iqa_results = self._validate_core(lr, hr)
            
            validation_results = self.iqa_evaluator.cache_out()
            if self.wandb_run is not None:
                self.wandb_run.log(validation_results, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
            
            # select only one image to display
            if self.config.wandb_validation_image_visualization:
                if self.wandb_run is not None:
                    b, c, h, w = sr.shape
                    sr = to_pil_image(sr[random.randint(0, b - 1), ...])
                    sr = wandb.Image(sr, caption="sr output")
                    self.wandb_run.log({"SR outputs": [sr]}, step=self.train_flags.curr_iter * self.wandb_run.config.batch_scale)
        
        self.generator.train()
        return validation_results
    
    def _validate_core(self, lr, hr):
        
        gen = self.generator
        if isinstance(gen, DP):
            gen = gen.module
        
        gen.eval()
        sr = gen(lr)
        
        sr = modcrop(sr, self.config.valid_scale_factor)
        hr = modcrop(hr, self.config.valid_scale_factor)
        iqa_results = self.iqa_evaluator(sr, hr, metrics=self.config.METRICS)
        
        return sr, iqa_results
    
    def run(self):
        
        if torch.cuda.device_count() > 1:
            print("Using DataParallel")
            self.generator = self.generator.cpu()
            self.generator = DP(self.generator)
            
        
        self.generator = self.generator.cuda()
        while not self.train_flags.STOP:
            self.train_flags.tic()
            
            self.trainval_1epoch()
            
            self.train_flags.toc()
            self.train_flags.print_times()
        
        self.post_process()


class TripletTrainer(DefaultTrainer):
    def __init__(self, *args, **kwargs):
        super(TripletTrainer, self).__init__(*args, **kwargs)

    def train_1iter(self, data_batch):
        self.optimizer.zero_grad()
    
        lr = data_batch["LR"].to(self.config.device)
        hr = data_batch["HR"].to(self.config.device)  # this is actually pretrained SR
        aux = data_batch["AUX"].to(self.config.device)  # this is real HR
    
    
        self._train_core(lr, hr, aux=aux)
    
        self.engine_interrupt_per_iter(self)
        self.train_flags.step()  # increase curr_iter, set self.train_flags.STOP
    
    def _train_core(self, lr, hr, aux):
        sr = self.generator(lr)
        loss = self.loss_modules(sr, hr, aux=aux)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

