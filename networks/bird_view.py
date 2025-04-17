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
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1), nn.BatchNorm2d(cout), backbone.act_layer
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1), nn.BatchNorm2d(cout), backbone.act_layer
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
grid_2_point_scale_05 = backbone.BilinearSample(scale_rate=(0.5, 0.5))
grid_2_point_scale_025 = backbone.BilinearSample(scale_rate=(0.25, 0.25))
grid_2_point_scale_0125 = backbone.BilinearSample(scale_rate=(0.125, 0.125))


class BEVNet(nn.Module):
    def __init__(self):
        super(BEVNet, self).__init__()

        # ----- Header -----
        self.cart_header = self._make_layer(backbone.BasicBlock, 192, 32, num_blocks=2)
        self.polar_header = self._make_layer(backbone.BasicBlock, 192, 32, num_blocks=2)

        # ----- Res1Block -----
        self.cart_res1 = self._make_layer(backbone.BasicBlock, 32, 64, num_blocks=3)
        self.polar_res1 = self._make_layer(backbone.BasicBlock, 32, 64, num_blocks=3)

        # ----- Res2Block -----
        self.cart_res2 = self._make_layer(backbone.BasicBlock, 64, 128, num_blocks=4)
        self.polar_res2 = self._make_layer(backbone.BasicBlock, 64, 128, num_blocks=4)

        # ----- Up2Block -----
        self.cart_up2 = AttMerge(64, 128, 96, scale_factor=2)
        self.polar_up2 = AttMerge(64, 128, 96, scale_factor=2)

        # ----- Up1Block -----
        self.cart_up1 = AttMerge(32, 96, 64, scale_factor=2)
        self.polar_up1 = AttMerge(32, 96, 64, scale_factor=2)

        self.out_channels = 64

        self.c_conv_1 = backbone.BasicConv2d(224, 128, kernel_size=3, padding=1)
        self.c_conv_2 = backbone.BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)
        self.p_conv_1 = backbone.BasicConv2d(224, 128, kernel_size=3, padding=1)
        self.p_conv_2 = backbone.BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)

        self.c_aux_0 = nn.Conv2d(32, 3, 1)
        self.c_aux_1 = nn.Conv2d(64, 3, 1)
        self.c_aux_2 = nn.Conv2d(128, 3, 1)

        self.p_aux_0 = nn.Conv2d(32, 3, 1)
        self.p_aux_1 = nn.Conv2d(64, 3, 1)
        self.p_aux_2 = nn.Conv2d(128, 3, 1)

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

    def forward(self, c, p, c_coord, p_coord):
        """
        c : [BS, 192, 512, 512]
        p : [BS, 192, 512, 512]
        c_coord : [BS, 160000, 2, 1]
        p_coord : [BS, 160000, 2, 1]
        """

        # Header 단계
        c0 = self.cart_header(c)  # [BS, 32, 256, 256]
        c1 = self.cart_res1(c0)  # [BS, 64, 128, 128]
        c1_point = grid_2_point_scale_025(c1, c_coord)
        c2 = self.cart_res2(c1)  # [BS, 128, 64, 64]

        p0 = self.polar_header(p)  # [BS, 32, 256, 256]
        p1 = self.polar_res1(p0)  # [BS, 64, 128, 128]
        p1_point = grid_2_point_scale_025(p1, p_coord)
        p2 = self.polar_res2(p1)  # [BS, 128, 64, 64]

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
        c_res_0 = F.interpolate(c0, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 32, 256, 256]
        c_res_1 = F.interpolate(c1, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 64, 256, 256]
        c_res_2 = F.interpolate(c2, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 128, 256, 256]
        c_res = [c_res_0, c_res_1, c_res_2]
        c_out = torch.cat(c_res, dim=1)  # [BS, 224, 256, 256]
        c_out = self.c_conv_2(self.c_conv_1(c_out))  # [BS, 3, 256, 256]
        c_out_point = grid_2_point_scale_05(c_out, c_coord)
        c_res_0 = self.c_aux_0(c_res_0)
        c_res_1 = self.c_aux_1(c_res_1)
        c_res_2 = self.c_aux_2(c_res_2)

        p_res_0 = F.interpolate(p0, size=p0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 32, 256, 256]
        p_res_1 = F.interpolate(p1, size=p0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 64, 256, 256]
        p_res_2 = F.interpolate(p2, size=p0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 128, 256, 256]
        p_res = [p_res_0, p_res_1, p_res_2]
        p_out = torch.cat(p_res, dim=1)  # [BS, 224, 256, 256]
        p_out = self.p_conv_2(self.p_conv_1(p_out))  # [BS, 3, 256, 256]
        p_out_point = grid_2_point_scale_05(p_out, p_coord)
        p_res_0 = self.p_aux_0(p_res_0)
        p_res_1 = self.p_aux_1(p_res_1)
        p_res_2 = self.p_aux_2(p_res_2)

        return (
            (c_out_point, p_out_point),
            (c1_point, p1_point),
            (c_res_0, p_res_0),
            (c_res_1, p_res_1),
            (c_res_2, p_res_2),
        )
