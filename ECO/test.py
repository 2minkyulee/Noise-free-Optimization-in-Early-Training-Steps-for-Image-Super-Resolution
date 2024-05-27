import tqdm
from utils.misc_utils import load_config_template, get_n_params, isNullStr, printl
import os
import random
import time
import argparse
import wandb
import torch
import numpy as np
from utils.train_utils import define_loader, define_optimizer, define_scheduler, CUDAPrefetcher
from utils.iqa_evaluate_utils import IQA_evaluator
import importlib
from multiprocessing import Process
from utils.img_proc_utils import imsave, modcrop, bicubic_resize
import utils.img_proc_utils


def main(wandb_run):

    # define generator(=network)
    g = wandb_run.config.generator  # generator type. (ESRGAN|SwinIR ... etc)
    g1, g2 = g.split("@")
    generator = getattr(importlib.import_module(f"model_arch.{g1}_arch"), g2)()
    generator = generator.to(wandb_run.config.device)

    # define dataloader
    test_loader = define_loader(mode="test", config=wandb_run.config)
    
    # load generator
    print(f"loading generator from {wandb_run.config.load_path_generator}")
    generator.load_state_dict_wrapper(torch.load(wandb_run.config.load_path_generator))
    
    # @tag [Test Loop]
    os.makedirs(wandb_run.config.save_path_test, exist_ok=True)
    with open(os.path.join(wandb_run.config.save_path_test, "log.txt"), "w") as log:
        test(wandb_run,
             test_loader=test_loader,
             generator=generator,
             log=log
             )
    wandb_run.finish(quiet=True)
    


def test(wandb_run,
         test_loader,
         generator,
         log
         ):
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    device = wandb_run.config.device
    generator.eval()
   
   
    # eval details
    crop = wandb_run.config.crop if wandb_run.config.crop>=0 else wandb_run.config.test_scale_factor
    mod = args.mod
    quantize = not wandb_run.config.no_quantize
    y_channel = not wandb_run.config.on_rgb
    
    iqa_evaluator = IQA_evaluator(
        metrics=["PSNR", "SSIM", "NIQE", "LPIPS"],
        crop_border=crop,  # if given -1, crop as much as scale
        test_y_channel=y_channel,
        do_quantize=quantize
    )

    printl(f"N_PARAMS: {get_n_params(generator)}", file=log)


    with torch.no_grad():

        # Initialize the data loader and load the first batch of data
        processes = []
        n_samples = 0
        for data_batch in tqdm.tqdm(test_loader):
            
            torch.cuda.empty_cache()
            lr = data_batch["LR"].to(device)
            hr = data_batch["HR"].to(device)
            sr = generator(lr)
            

            if not isNullStr(wandb_run.config.save_path_test):
                p = Process(target=imsave,
                            kwargs={
                                "imgs":sr.cpu(),
                                "names":data_batch["name"],
                                "save_dir":os.path.join(wandb_run.config.save_path_test, "images"),
                                "verbose": True
                            })
                p.start()
                processes.append(p)
            if len(processes)>100:
                for p in processes:
                    p.join()
                    p.close()
                processes = []
            
            n_samples += 1
            
            sr = modcrop(sr, mod)
            hr = modcrop(hr, mod)
            eval_result_one_image = iqa_evaluator(sr.cuda(), hr.cuda(), metrics=["PSNR", "SSIM"], verbose=False)
            print(data_batch["name"], dict((k, v.item()) for k, v in eval_result_one_image.items()), file=log)
        
        
        # ----------------------------- end test, now save/log ------------------ #
        
        eval_result_total = iqa_evaluator.cache_out()
        printl("-----------------------------------------", file=log)
        printl(f"gen_arch: {wandb_run.config.generator}", file=log)
        printl(f"gen_path: {wandb_run.config.load_path_generator}", file=log)
        printl(f"HR_path: {wandb_run.config.test_path_imageHR}", file=log)
        printl(f"LR_path: {wandb_run.config.test_path_imageLR}", file=log)
        printl(f"scale: {wandb_run.config.test_scale_factor}", file=log)
        printl(f"crop: {crop}", file=log)
        printl(f"mod: {mod}", file=log)
        printl(f"on_y_channel: {y_channel}", file=log)
        printl(f"quantize: {quantize}", file=log)
        
        results = []
        _ = tuple((results.append(k), results.append(v)) for k, v in eval_result_total.items())
        printl(*results, sep=",", file=log)
        printl(file=log)
        printl(f"{results[1]}, {results[3]}", file=log)
        
        for p in processes:
            p.join()
            p.close()

if __name__=="__main__":

    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--project_name")
    argparser.add_argument("--generator")
    argparser.add_argument("--load_path_generator")
    argparser.add_argument("--save_path_test")
    argparser.add_argument("--test_path_imageHR", default="")
    argparser.add_argument("--test_path_imageLR", default="")
    argparser.add_argument("--test_path_imageAUX", default="")
    argparser.add_argument("--test_scale_factor", type=int)
    argparser.add_argument("--no_quantize", action="store_true", default=False)
    argparser.add_argument("--on_rgb", action="store_true", default=False)
    argparser.add_argument("--crop", type=int, default=-1)   # border crop when evaluation. if set as -1, crop=test_scale_factor
    argparser.add_argument("--mod", type=int)    # which type of GT to use. (ex: GTmod12) / use scale_factor for default
    argparser.add_argument("--debug", default=False, action="store_true")
    args = argparser.parse_args()

    config_template = load_config_template("test_base_configs", args)

    wandb_run = wandb.init(
        name=args.project_name,     # wont use it anyway
        project=args.project_name,  # wont use it anyway
        mode="disabled"
    )
    wandb_run.config.update(vars(config_template), allow_val_change=True)
    wandb_run.config.update(args, allow_val_change=True)  # config priority 1st
    wandb_run.config.proj_start_time = time.time()

    main(wandb_run=wandb_run)
    wandb_run.finish(quiet=True)
    