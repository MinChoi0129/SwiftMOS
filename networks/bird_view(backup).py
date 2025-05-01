import math
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import deep_point
from utils.polar_cartesian import Cart2Polar, Polar2Cart
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

        x_high_up = F.interpolate(
            x_high, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        x_merge = torch.stack(
            (self.conv_low(x_low), self.conv_high(x_high_up)), dim=1
        )  # (BS, 2, channels, H, W)
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
c2p_128 = Cart2Polar(polar_size=(128, 128), cart_size=(128, 128))
p2c_128 = Polar2Cart(polar_size=(128, 128), cart_size=(128, 128))
p2c_256 = Polar2Cart(polar_size=(256, 256), cart_size=(256, 256))


class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()

        # ---------- SA 인코더 ----------
        self.cart_sa = backbone.ViTEncoder(in_ch=192)
        self.polar_sa = backbone.ViTEncoder(in_ch=192)

        # ----- Header -----
        self.cart_header = self._make_layer(backbone.BasicBlock, 192, 32, num_blocks=2)
        self.polar_header = self._make_layer(backbone.BasicBlock, 192, 32, num_blocks=2)

        # ----- ResBlock1 -----
        self.cart_res1 = self._make_layer(backbone.BasicBlock, 32, 64, num_blocks=3)
        self.polar_res1 = self._make_layer(backbone.BasicBlock, 32, 64, num_blocks=3)
        self.cart_res1_fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.polar_res1_fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ----- ResBlock2 -----
        self.cart_res2 = self._make_layer(backbone.BasicBlock, 64, 128, num_blocks=4)
        self.polar_res2 = self._make_layer(backbone.BasicBlock, 64, 128, num_blocks=4)

        # ----- UpBlock2 -----
        self.cart_up2 = AttMerge(64, 128, 96, scale_factor=2)
        self.polar_up2 = AttMerge(64, 128, 96, scale_factor=2)
        self.cart_up2_fuse = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.polar_up2_fuse = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # ----- UpBlock1 -----
        self.cart_up1 = AttMerge(32, 96, 64, scale_factor=2)
        self.polar_up1 = AttMerge(32, 96, 64, scale_factor=2)

    def _make_layer(
        self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1
    ):
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

    def fuse(self, cart, polar, mode):
        cart_fuse = torch.cat((p2c_128(polar, cart), cart), dim=1)
        polar_fuse = torch.cat((c2p_128(cart, polar), polar), dim=1)
        if mode == "res1":
            c1 = self.cart_res1_fuse(cart_fuse)
            p1 = self.polar_res1_fuse(polar_fuse)
            return c1, p1
        elif mode == "up2":
            c3 = self.cart_up2_fuse(cart_fuse)
            p3 = self.polar_up2_fuse(polar_fuse)
            return c3, p3
        else:
            raise ValueError("Invalid mode. Choose 'res1' or 'up2'.")

    def forward(self, c, p):
        """
        c : [BS, 192, 512, 512]
        p : [BS, 192, 512, 512]
        c_coord : [BS, 160000, 2, 1]
        p_coord : [BS, 160000, 2, 1]
        """

        # Self Attention 단계
        c = self.cart_sa(c)  # [BS, 192, 512, 512]
        p = self.polar_sa(p)  # [BS, 192, 512, 512]

        # Header 단계
        c0 = self.cart_header(c)  # [BS, 32, 256, 256]
        p0 = self.polar_header(p)  # [BS, 32, 256, 256]

        c1 = self.cart_res1(c0)  # [BS, 64, 128, 128]
        p1 = self.polar_res1(p0)  # [BS, 64, 128, 128]
        c1, p1 = self.fuse(c1, p1, mode="res1")

        c2 = self.cart_res2(c1)  # [BS, 128, 64, 64]
        p2 = self.polar_res2(p1)  # [BS, 128, 64, 64]

        c3 = self.cart_up2(c1, c2)  # [BS, 96, 128, 128]
        p3 = self.polar_up2(p1, p2)  # [BS, 96, 128, 128]
        c3, p3 = self.fuse(c3, p3, mode="up2")

        c4 = self.cart_up1(c0, c3)  # [BS, 64, 256, 256]
        p4 = self.polar_up1(p0, p3)  # [BS, 64, 256, 256]
        cart_up1_fuse = torch.cat((p2c_256(p4, c4), c4), dim=1)  # [BS, 128, 256, 256]

        return cart_up1_fuse  # [BS, 128, 256, 256]
