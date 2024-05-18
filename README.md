# [AAAI2024] Noise-free Optimization in Early Training Steps for Image Super-Resolution
Official Repository for AAAI2024 Noise-free Optimization in Early Training Steps for Image Super-Resolution (ECO). \
[ArXiv] https://arxiv.org/abs/2312.17526 \
[AAAI] https://doi.org/10.1609/aaai.v38i4.28073

---

## Abstract
Recent deep-learning-based single image super-resolution (SISR) methods have shown impressive performance whereas typical methods train their networks by minimizing the pixel-wise distance with respect to a given high-resolution (HR) image. However, despite the basic training scheme being the predominant choice, its use in the context of ill-posed inverse problems has not been thoroughly investigated. In this work, we aim to provide a better comprehension of the underlying constituent by decomposing target HR images into two subcomponents: (1) the optimal centroid which is the expectation over multiple potential HR images, and (2) the inherent noise defined as the residual between the HR image and the centroid. Our findings show that the current training scheme cannot capture the ill-posed nature of SISR and becomes vulnerable to the inherent noise term, especially during early training steps. To tackle this issue, we propose a novel optimization method that can effectively remove the inherent noise term in the early steps of vanilla training by estimating the optimal centroid and directly optimizing toward the estimation. Experimental results show that the proposed method can effectively enhance the stability of vanilla training, leading to overall performance gain.

---
## Notes

:star:**Our implementation is based on DP (DataParallel).** \
DP is reported to be faster than DDP without NVlinks. (https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_many)

:star:**Weights are compatible with official repositories of each, not BasicSR.** \
EDSR: https://github.com/sanghyun-son/EDSR-PyTorch \
RCAN: https://github.com/yulunzhang/RCAN

:exclamation: **Notes on Downsampling with MATLAB** \
In official repositories, there is variation in how scripts generate bicubic downsampled images: some utilize the **im2double** function to increase precision, while others do not. Given that SR methods may be sensitive to such differences, we have manually downsampled all train/test images **_without_** using the **im2double** function, which is the de facto configuration. Both our models and the baseline models have been trained and tested using this consistently downsampled dataset.
The BasicSR version of our code will utilize the **im2double** function, aligning with the default configuration of the BasicSR framework.


:eyes: **Support and Feedback.** \
We are currently refining/cleaning-up our codebase. Should you encounter any issues or bugs, please do not hesitate to contact us. Your feedback is invaluable to our ongoing efforts to improve our work.


---

## Todo
:star:**Implementation based on BasicSR.** \
Currently, the codes are mostly implemented from scratch. We are in the process of adapting our implementation to integrate with BasicSR.


---


## Overall Framework of ECO
![teaser](assets/main_figure.png)


---


## Quantitative Results
![maintable](assets/eco_quant_table.png)



---
## Pretrained Weights
Download pretrained weights of reproduced baselines and ours [here](https://drive.google.com/drive/folders/1JY_mnH780kKV9iEGcFDfa_dlvBEu-fKf?usp=sharing).


---
## Basic Usage

#### Example script for training
```
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 train.py --wandb_name ECOO --config_template _largerbatch_EDSR_S_x2_ecoo2_mixup_lrx2_configs  # reproducing ours
CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python3 train.py --wandb_name ECOO --config_template _largerbatch_EDSR_x2_default_configs             # reproducing baseline
```

#### Example script for training with DP
```
# Providing more than 1 device will automatically start DP.
CUDA_VISIBLE_DEVICES=0,1 /opt/conda/bin/python3 train.py --wandb_name ECOO --config_template your_config  
```

#### Example script to resume training
```
# Manually provide wandb_id. If not, acts as normal training (without resuming).
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 train.py --wandb_name ECOO --config_template your_config_name --wandb_id your_wandb_run_id
```

#### Example script for testing x3 SR with RCAN. 
```
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 test_official.py \
--project_name RCAN_x3  \
--generator RCAN@RCAN_x3  \
--test_scale_factor 3  \
--mod 3  \
--save_path_test /YOUR_SAVE_PATH \
--test_path_imageHR /YOUR_DATA_PATH/sr_dataset_fp32/Urban100/HR \
--test_path_imageLR /YOUR_DATA_PATH/sr_dataset_fp32/Urban100/LRbicx3_fp32 \
--load_path_generator /YOUR_WEIGHT_PATH/rcan_x3_default_batchx8_cosine_best.pth \  # or any other RCAN_x3 weight, including ours. 
```
#### Example script for testing x2 SR with EDSR. 
```
CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python3 test_official.py \
--project_name EDSR_x2  \
--generator EDSR@EDSR_x2  \
--test_scale_factor 2  \
--mod 2  \
--save_path_test /YOUR_SAVE_PATH \
--test_path_imageHR /YOUR_DATA_PATH/sr_dataset_fp32/Urban100/HR \
--test_path_imageLR /YOUR_DATA_PATH/sr_dataset_fp32/Urban100/LRbicx2_fp32 \
--load_path_generator /YOUR_WEIGHT_PATH/edsr_x2_ECO2_batchx8_cosine_best.pth \  # or any other EDSR_x2 weight, including baseline.
```

---
## Acknowledgement
Our code is based on the following repositories. \
https://github.com/XPixelGroup/BasicSR \
https://github.com/sanghyun-son/EDSR-PyTorch \
https://github.com/yulunzhang/RCAN \
https://github.com/XPixelGroup/HAT \
https://github.com/cszn/KAIR/ \
https://github.com/JingyunLiang/SwinIR \
https://github.com/kligvasser/SANGAN \
https://github.com/chaofengc/IQA-PyTorch/


---


## Citation
Consider citing us if you find our paper useful in your research :smile:
```
@inproceedings{lee2024noise,
  title={Noise-free optimization in early training steps for image super-resolution},
  author={Lee, MinKyu and Heo, Jae-Pil},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={2920--2928},
  year={2024}
}
```

---


## Contact
Please contact me via 2minkyulee@gmail.com for any inquiries.

