"""
Modified from BasicSR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction  import create_feature_extractor, get_graph_node_names
import torchvision.models as models
from torchvision import transforms


class PerceptualLoss(nn.Module):

    def __init__(self, loss_logger):
        super(PerceptualLoss, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # vgg.load_state_dict(torch.load("/mnt/hdd0/mklee/pretrained_weights/vgg/vgg19.pth"))

        self.layer_coef = {
            "features.2": 0.1,
            "features.7": 0.1,
            "features.16": 1,
            "features.25": 1,
            "features.34": 1,
        }
        self.extraction_layer = [
            "features.2",
            "features.7",
            "features.16",
            "features.25",
            "features.34",
        ]



        self.vgg = create_feature_extractor(vgg, self.extraction_layer)
        self.vgg.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # freeze
        for params in self.vgg.parameters():
            params.requires_grad = False

        self.loss_logger = loss_logger




    def forward(self, sr, hr, coef, **kwargs):

        sr = self.normalize(sr)
        hr = self.normalize(hr)

        perceptual_loss = torch.tensor([0], device="cuda", dtype=torch.float32)
        for layer in self.extraction_layer:
            sr_feat = self.vgg(sr)[layer]
            hr_feat = self.vgg(hr)[layer]
            layer_coef = self.layer_coef[layer]

            perceptual_loss += layer_coef * F.l1_loss(sr_feat, hr_feat)
        
        self.loss_logger.cache_in("Train/Loss/percep", perceptual_loss.item())

        return coef * perceptual_loss


# def sliced_wasserstein(self, x, y):
#
#     def generate_random_direction(shape_, n_direct=None):
#         """
#         :param shape: BCHW
#         :param n_direct: adaptive(=C) or fixed value
#         :return: n_direct x C
#         """
#         b,c,h,w = shape_
#
#         if n_direct is None:
#             n_direct = c
#
#         rand_direct = torch.randn(size=(n_direct, c)).cuda()
#         norm = torch.sqrt(torch.sum(torch.square(rand_direct), dim=-1))
#         norm = norm.view(n_direct, 1)
#         rand_direct = rand_direct/norm  # n_direct개의 (cx1) direction이 있음.
#
#         return rand_direct
#
#     def projection(f, rand_direct):
#
#         """
#         :param rand_direct: n_direct x C
#         :param f: B x C x H x W
#         :return: n_direct x sort(HW)
#         """
#         b, c1, h, w = f.shape
#         n_direct, c2 = rand_direct.shape
#         assert c1 == c2
#
#
#         f = f.view(b, c1, h*w) # b x c x hw
#
#         projected = rand_direct.matmul(f)  # (n_direct, c2) @ (b, c1, HW) --> (b, n_direct, HW) with implicit broadcast
#         projected, _ = torch.sort(projected, dim=-1)  # (b, n_direct, HW)
#
#         return projected
#
#     y = torch.cat((y, y, y, y), dim=3)
#     rand_direct = generate_random_direction(x.shape)
#     proj_x = projection(x, rand_direct)  # b x n_direct x sorted(HW)
#     proj_y = projection(y, rand_direct)
#
#     _, _, hw = proj_x.shape
#     proj_x = proj_x[:, :, :int(hw*0.9)]
#     proj_y = proj_y[:, :, :int(hw*0.9)]
#
#     return proj_x, proj_y
