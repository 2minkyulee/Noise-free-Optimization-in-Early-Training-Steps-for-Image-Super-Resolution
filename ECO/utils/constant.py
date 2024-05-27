"""
We define the paths to the datasets and the wandb directory in this file.
This setup allows for auto-completion of paths when using IDEs and easy management of dataset directories.

Modify the paths to the datasets and the wandb directory as needed.
"""


WANDB_PATH = "/mnt/hdd0/mklee/wandb"  # where wandb logs are saved
BASE_PATH = "/mnt/hdd0/mklee"  # where experiments are saved

# ----------------------- Train Set --------------------------------%

# original dataset
DATASET_HR_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/HR"
DATASET_LRBICx2_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain/LRbicx4_fp32_sub"


# ecoo_swap_EDSR_S_x2 dataset
DATASET_HR_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/HR"
DATASET_LRBICx2_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_swap_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_swap_EDSR_S_x2/LRbicx4_fp32_sub"

# ecoo_EDSR_S_x2 dataset
DATASET_HR_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/HR"
DATASET_LRBICx2_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_EDSR_S_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x2/LRbicx4_fp32_sub"


# ecoo_EDSR_S_x4 dataset
DATASET_HR_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/HR"
DATASET_LRBICx2_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_EDSR_S_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_S_x4/LRbicx4_fp32_sub"


# ecoo_EDSR_x2 dataset
DATASET_HR_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/HR"
DATASET_LRBICx2_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_EDSR_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x2/LRbicx4_fp32_sub"


# ecoo_EDSR_x3 dataset
DATASET_HR_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/HR"
DATASET_LRBICx2_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_EDSR_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x3/LRbicx4_fp32_sub"




# ecoo_EDSR_x4 dataset
DATASET_HR_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/HR"
DATASET_LRBICx2_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_EDSR_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_EDSR_x4/LRbicx4_fp32_sub"






# ecoo_RCAN_x2 dataset
DATASET_HR_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/HR"
DATASET_LRBICx2_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_RCAN_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x2/LRbicx4_fp32_sub"


# ecoo_RCAN_x3 dataset
DATASET_HR_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/HR"
DATASET_LRBICx2_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_RCAN_x3 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x3/LRbicx4_fp32_sub"




# ecoo_RCAN_x4 dataset
DATASET_HR_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/HR"
DATASET_LRBICx2_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_RCAN_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RCAN_x4/LRbicx4_fp32_sub"








# ecoo_RRDB_x2 dataset
DATASET_HR_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/HR"
DATASET_LRBICx2_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/HR_sub"
PREPROCESSED_DATASET_LRBICx2_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx2_fp32_sub"
PREPROCESSED_DATASET_LRBICx3_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx3_fp32_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_RRDB_x2 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_RRDB_x2/LRbicx4_fp32_sub"




# ecoo_HAT_x4 dataset
DATASET_HR_DIV2Ktrain_HAT_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_HAT_x4/HR"
DATASET_LRBICx4_DIV2Ktrain_HAT_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_HAT_x4/LRbicx4_fp32"

PREPROCESSED_DATASET_HR_DIV2Ktrain_HAT_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_HAT_x4/HR_sub"
PREPROCESSED_DATASET_LRBICx4_DIV2Ktrain_HAT_x4 = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Ktrain_HAT_x4/LRbicx4_fp32_sub"







# ----------------------- Test Set --------------------------------%


DATASET_HR_DIV2Kvalid = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kvalid/HR"
DATASET_LRBICx2_DIV2Kvalid = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kvalid/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Kvalid = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kvalid/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Kvalid = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kvalid/LRbicx4_fp32"

DATASET_HR_DIV2Kmini = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kmini/HR"
DATASET_LRBICx2_DIV2Kmini = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kmini/LRbicx2_fp32"
DATASET_LRBICx3_DIV2Kmini = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kmini/LRbicx3_fp32"
DATASET_LRBICx4_DIV2Kmini = f"/mnt/hdd0/mklee/sr_dataset_fp32/DIV2Kmini/LRbicx4_fp32"

DATASET_HR_Set5 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set5/HR"
DATASET_LRBICx2_Set5 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set5/LRbicx2_fp32"
DATASET_LRBICx3_Set5 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set5/LRbicx3_fp32"
DATASET_LRBICx4_Set5 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set5/LRbicx4_fp32"

DATASET_HR_Set14 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set14/HR"
DATASET_LRBICx2_Set14 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set14/LRbicx2_fp32"
DATASET_LRBICx3_Set14 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set14/LRbicx3_fp32"
DATASET_LRBICx4_Set14 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Set14/LRbicx4_fp32"

DATASET_HR_Urban100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Urban100/HR"
DATASET_LRBICx2_Urban100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Urban100/LRbicx2_fp32"
DATASET_LRBICx3_Urban100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Urban100/LRbicx3_fp32"
DATASET_LRBICx4_Urban100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Urban100/LRbicx4_fp32"

DATASET_HR_BSD100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/BSD100/HR"
DATASET_LRBICx2_BSD100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/BSD100/LRbicx2_fp32"
DATASET_LRBICx3_BSD100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/BSD100/LRbicx3_fp32"
DATASET_LRBICx4_BSD100 = f"/mnt/hdd0/mklee/sr_dataset_fp32/BSD100/LRbicx4_fp32"

DATASET_HR_Manga109 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Manga109/HR"
DATASET_LRBICx2_Manga109 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Manga109/LRbicx2_fp32"
DATASET_LRBICx3_Manga109 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Manga109/LRbicx3_fp32"
DATASET_LRBICx4_Manga109 = f"/mnt/hdd0/mklee/sr_dataset_fp32/Manga109/LRbicx4_fp32"



