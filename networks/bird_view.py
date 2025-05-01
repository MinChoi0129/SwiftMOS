import math
import os
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import deep_point
from utils.pretty_print import shprint
from . import backbone


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.contiguous().float(),
        pcds_ind=pcds_ind.contiguous(),
        output_size=output_size,
        scale_rate=scale_rate,
    ).to(pcds_feat.dtype)
    return voxel_feat


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.att_layer = nn.Sequential(
            backbone.conv3x3(2 * cout, cout // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cout // 2),
            backbone.act_layer,
            backbone.conv3x3(cout // 2, 2, stride=1, dilation=1, bias=True),
        )

        self.conv_high = nn.Sequential(
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer,
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer,
        )

    def forward(self, x_low, x_high):
        # shprint("Low High", x_low, x_high)
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        x_high_up = F.interpolate(x_high, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1)  # (BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # Attention 기반 융합
        ca_map = self.att_layer(x_merge.view(batch_size, 2 * self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)
        x_out = (x_merge * ca_map).sum(dim=1)  # (BS, channels, H, W)
        return x_out


"""
VoxelMaxPool : 두번째 파라미터가 갖는 값 quan 기준 W/H * scale_rate = output_size.
BilinearSample : 두번째 파라미터가 갖는 값 quan 기준 W/H가 첫번째 파라미터로 되기 위한 scale_rate.
"""

grid_2_point_scale_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))
grid_2_point_scale_05 = backbone.BilinearSample(scale_rate=(0.5, 0.5))
grid_2_point_scale_025 = backbone.BilinearSample(scale_rate=(0.25, 0.25))
bev_scale_rates = {
    512: (1.0, grid_2_point_scale_full),
    256: (0.5, grid_2_point_scale_05),
    128: (0.25, grid_2_point_scale_025),
}
polar_scale_rates = {
    64: (1.0, grid_2_point_scale_full),
    32: (0.5, grid_2_point_scale_05),
    16: (0.25, grid_2_point_scale_025),
}


class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()

        # ----- Header -----
        self.cart_header = self._make_layer(backbone.BasicBlock, 192, 32, 2)
        self.polar_header = self._make_layer(backbone.BasicBlock, 32, 32, 1, stride=1)

        # ----- ResBlock1 -----
        self.cart_res1 = self._make_layer(backbone.BasicBlock, 64, 64, 3)
        self.polar_res1 = self._make_layer(backbone.BasicBlock, 64, 64, 2, stride=1)

        # ----- ResBlock2 -----
        self.cart_res2 = self._make_layer(backbone.BasicBlock, 128, 128, 4)

        # ───────── Temporal fusion (Addition) ─────────
        self.add_fuse = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_conv1 = backbone.BasicConv2d(320, 128, kernel_size=3, padding=1)
        self.out_conv2 = backbone.BasicConv2d(128, 64, kernel_size=3, padding=1)

        self.aux_head1 = nn.Conv2d(64, 3, 1)
        self.aux_head2 = nn.Conv2d(128, 3, 1)
        self.aux_head3 = nn.Conv2d(128, 3, 1)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    @staticmethod
    def save_feature_as_img(variable, variable_name):
        save_dir = f"/home/workspace/work/TripleMOS/images/{variable_name}"
        os.makedirs(save_dir, exist_ok=True)  # 🔧 폴더가 없으면 생성

        single_batch = variable[0].cpu().numpy()
        for i, c in enumerate(single_batch):
            plt.imsave(f"{save_dir}/{i:06}.png", c)

    def c2p(self, c, c_coord_curr, p_coord_curr):
        BS, C, H, W = c.shape

        scale_rate, grid_to_point = bev_scale_rates[H]
        point = grid_to_point(c, c_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=p_coord_curr,
            output_size=(int(64 * scale_rate), int(2048 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def p2c(self, p, c_coord_curr, p_coord_curr):
        BS, C, H, W = p.shape

        scale_rate, grid_to_point = polar_scale_rates[H]
        point = grid_to_point(p, p_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=c_coord_curr,
            output_size=(int(512 * scale_rate), int(512 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def forward(self, c, c_coord_curr, p_coord_curr, deep_64):
        """
        c : [BS, 192, 512, 512]
        c_coord_curr : [BS, 160000, 2, 1]
        p_coord_curr : [BS, 160000, 2, 1]
        deep_64 : [BS, 128, 64, 64]
        """

        # --------------- Cart ---------------
        c0 = self.cart_header(c)  # [BS, 32, 256, 256]
        c0_to_polar = self.c2p(c0, c_coord_curr, p_coord_curr)  # [BS, 32, 32, 1024]

        p0 = self.polar_header(c0_to_polar)  # [BS, 32, 32, 1024]
        p0_to_cart = self.p2c(p0, c_coord_curr, p_coord_curr)  # [BS, 32, 256, 256]

        c0 = torch.cat((c0, p0_to_cart), dim=1)  # [BS, 64, 256, 256]

        c1 = self.cart_res1(c0)  # [BS, 64, 128, 128]
        c1_to_polar = self.c2p(c1, c_coord_curr, p_coord_curr)  # [BS, 64, 16, 512]

        p1 = self.polar_res1(c1_to_polar)  # [BS, 64, 16, 512]
        p1_to_cart = self.p2c(p1, c_coord_curr, p_coord_curr)  # [BS, 64, 128, 128]

        c1 = torch.cat((c1, p1_to_cart), dim=1)  # [BS, 128, 128, 128]

        c2 = self.cart_res2(c1)  # [BS, 128, 64, 64]

        # --------------- Temporal Memory ---------------
        if deep_64 is not None:
            fused = c2 + deep_64  #
            c2 = self.add_fuse(fused)

        # --------------- Decoder ---------------
        res0 = F.interpolate(c0, size=c0.size()[2:], mode="bilinear", align_corners=True)
        res1 = F.interpolate(c1, size=c0.size()[2:], mode="bilinear", align_corners=True)
        res2 = F.interpolate(c2, size=c0.size()[2:], mode="bilinear", align_corners=True)
        out = self.out_conv2(self.out_conv1(torch.cat([res0, res1, res2], dim=1)))  # [BS, 64, 256, 256]

        # --------------- BackProject ---------------
        out_as_point = grid_2_point_scale_05(out, c_coord_curr)
        polar_res1_as_point = grid_2_point_scale_05(p1, p_coord_curr)

        # --------------- Aux Head ---------------
        res0 = self.aux_head1(res0)
        res1 = self.aux_head2(res1)
        res2 = self.aux_head3(res2)

        return (
            out_as_point,  # (BS, 64, 160000, 1)
            polar_res1_as_point,  # (BS, 64, 160000, 1)
            res0,  # (BS, 3, 256, 256)
            res1,  # (BS, 3, 256, 256)
            res2,  # (BS, 3, 256, 256)
            c2,  # (BS, 128, 64, 64)
        )
