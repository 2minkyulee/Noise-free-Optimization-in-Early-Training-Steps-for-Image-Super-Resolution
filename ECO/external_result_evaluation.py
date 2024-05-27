from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import natsort
import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from utils.misc_utils import get_image_names, clear_path
from utils.img_proc_utils import center_crop, bicubic_resize, modcrop
from utils.iqa_evaluate_utils import IQA_evaluator


class ExternalValidationDataset(Dataset):
    def __init__(self, path_imageHR, path_imageSR, return_file_name=False):
        super(ExternalValidationDataset, self).__init__()
        self.hr_image_names = natsort.natsorted(get_image_names(path_imageHR))
        self.sr_image_names = natsort.natsorted(get_image_names(path_imageSR))
        self.len = len(self.hr_image_names)

        self.return_file_name = return_file_name

        assert len(self.hr_image_names)==len(self.sr_image_names),\
            f"got {len(self.hr_image_names)} images for HR dir, but {len(self.sr_image_names)} images for SR dir"
        

    def __getitem__(self, idx):
        # read
        sr_image = to_tensor(Image.open(self.sr_image_names[idx]))
        hr_image = to_tensor(Image.open(self.hr_image_names[idx]))

        # for set14
        # assert hr_image.shape == sr_image.shape, print(hr_image.shape, sr_image.shape, self.sr_image_names[idx])
        
        if self.return_file_name:
            return {
                "SR": sr_image,
                "HR": hr_image,
                "filename": (os.path.basename(self.hr_image_names[idx]), os.path.basename(self.sr_image_names[idx]))
            }
        
        else:
            return {"SR": sr_image,
                    "HR": hr_image
                    }

    def __len__(self):
        return self.len


    
def evaluate(path_imageHR, path_imageSR, scale, mod, on_rgb, do_quantize):
    
    dataset = ExternalValidationDataset(
        path_imageHR=path_imageHR,
        path_imageSR=path_imageSR,
        return_file_name=True,
    )
    
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=0,
        batch_size=1,
    )
    
    
    
    crop = scale
    mod = mod
    y_channel = not on_rgb
    print(f"crop: {crop}")
    print(f"scale: {scale}")
    print(f"mod: {mod}")
    print(f"y_channel: {y_channel}")
    print(f"quantize: {do_quantize}")
    
    
    iqa_evaluator = IQA_evaluator(
        metrics=["PSNR", "SSIM"],
        crop_border=crop,
        test_y_channel=y_channel,
        do_quantize=do_quantize,
    )

    
    for batch_data in dataloader:
        sr = batch_data["SR"].cuda()
        hr = batch_data["HR"].cuda()
        filename = batch_data["filename"]
    
        if hr.shape[-3]==1:
            hr = hr.repeat(1,3,1,1)
            
            sr = sr.mean(dim=1, keepdim=True)
            sr = sr.repeat(1,3,1,1)
            
        sr = modcrop(sr, mod)
        hr = modcrop(hr, mod)
        score = iqa_evaluator(sr, hr, metrics="ALL", verbose=False)
        print(filename)
        print(score)
        
        
    val_results = iqa_evaluator.cache_out()
    print(val_results)
    
if __name__ == "__main__":
    
    import argparse
    import warnings
    warnings.filterwarnings(action="ignore")
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_imageHR")
    argparser.add_argument("--path_imageSR")
    argparser.add_argument("--scale", type=int, default=4)
    argparser.add_argument("--mod", type=int, default=4)
    args = argparser.parse_args()
    
    evaluate(
        path_imageHR=args.path_imageHR,  #f"/mnt/hdd0/mklee/sr_dataset_fp32/Set14/HR",
        path_imageSR=args.path_imageSR,  #f"/mnt/hdd0/mklee/mklee_SR_framework/inference_results/RCAN_from_github/RCAN/Set14/x2",
        scale=args.scale,  # 4,
        mod=args.mod,  # 4,
        on_rgb=False,  # ycbcr
        do_quantize=True  #True
    )
