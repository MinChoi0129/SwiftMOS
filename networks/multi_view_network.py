import math
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import deep_point
from . import backbone
import numpy as np


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.contiguous().float(),
        pcds_ind=pcds_ind.contiguous(),
        output_size=output_size,
        scale_rate=scale_rate,
    ).to(pcds_feat.dtype)
    return voxel_feat


"""
VoxelMaxPool : 두번째 파라미터가 갖는 값 quan 기준 W/H * scale_rate = output_size.
BilinearSample : 두번째 파라미터가 갖는 값 quan 기준 W/H가 첫번째 파라미터로 되기 위한 scale_rate.
"""

# 변환에 쓰일 물리적 범위 (pModel.Voxel.* 값을 그대로 사용)
X_RANGE = (-50.0, 50.0)  # [m]
Y_RANGE = (-50.0, 50.0)  # [m]
R_RANGE = (2.0, 50.0)  # [m]
PHI_RANGE = (-180.0, 180.0)  # [deg]
THETA_RANGE = (-25.0, 3.0)  # [deg]

grid_2_point_scale_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))
grid_2_point_scale_05 = backbone.BilinearSample(scale_rate=(0.5, 0.5))
grid_2_point_scale_025 = backbone.BilinearSample(scale_rate=(0.25, 0.25))
grid_2_point_scale_0125 = backbone.BilinearSample(scale_rate=(0.125, 0.125))

descartes_scale_rates = {
    512: (1.0, grid_2_point_scale_full),
    256: (0.5, grid_2_point_scale_05),
    128: (0.25, grid_2_point_scale_025),
    64: (0.125, grid_2_point_scale_0125),
}
sphere_scale_rates = {
    64: (1.0, grid_2_point_scale_full),
    32: (0.5, grid_2_point_scale_05),
    16: (0.25, grid_2_point_scale_025),
    8: (0.125, grid_2_point_scale_0125),
}


class MultiViewNetwork(nn.Module):
    def __init__(self):
        super(MultiViewNetwork, self).__init__()

        # ----- Header -----
        self.descartes_header = self._make_layer(backbone.BasicBlock, 192, 32, 2)
        self.sphere_header = self._make_layer(backbone.BasicBlock, 32, 32, 1, stride=1)
        self.header_channel_down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        # ----- ResBlock1 -----
        self.descartes_res1 = self._make_layer(backbone.BasicBlock, 32, 64, 3)
        self.sphere_res1 = self._make_layer(backbone.BasicBlock, 64, 64, 2, stride=1)
        self.res1_channel_down = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # ----- ResBlock2 -----
        self.descartes_res2 = self._make_layer(backbone.BasicBlock, 64, 128, 4)

        # ───────── Temporal fusion (Addition) ─────────
        self.add_fuse = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_conv1 = backbone.BasicConv2d(32 + 64 + 128, 128, kernel_size=3, padding=1)
        self.out_conv2 = backbone.BasicConv2d(128, 64, kernel_size=3, padding=1)

        self.aux_head1 = nn.Conv2d(32, 3, 1)
        self.aux_head2 = nn.Conv2d(64, 3, 1)
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
        save_dir = f"images/features/{variable_name}"
        os.makedirs(save_dir, exist_ok=True)  # 🔧 폴더가 없으면 생성

        single_batch = variable[0].cpu().numpy()
        for i, c in enumerate(single_batch):
            plt.imsave(f"{save_dir}/{i:06}.png", c)

    @staticmethod
    def des_2_sph_direct(bev_feat, z_map):
        """
        bev_feat : (B, C, Hb, Wb)   z_map : (B, 1, Hb, Wb)
        반환        : (B, C, Hr, Wr)  (Hr = Hb//8, Wr = Wb*4)
        """
        B, C, Hb, Wb = bev_feat.shape
        Hr, Wr = Hb // 8, Wb * 4
        dev = bev_feat.device

        # 1. BEV 셀 중심 물리좌표 (x,y,z)
        u = torch.arange(Wb, device=dev).view(1, 1, 1, Wb)
        v = torch.arange(Hb, device=dev).view(1, 1, Hb, 1)
        x = X_RANGE[0] + (u + 0.5) / Wb * (X_RANGE[1] - X_RANGE[0])
        y = Y_RANGE[1] - (v + 0.5) / Hb * (Y_RANGE[1] - Y_RANGE[0])
        z = z_map  # (B,1,Hb,Wb)

        # 2. θ, φ, r 계산
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
        phi = torch.atan2(y, x)  # rad
        theta = torch.asin(z / r)  # rad

        # 3. RV 인덱스
        phi_min, phi_max = map(math.radians, PHI_RANGE)
        th_min, th_max = map(math.radians, THETA_RANGE)
        j = ((phi - phi_min) / (phi_max - phi_min) * (Wr - 1)).long()
        i = ((th_max - theta) / (th_max - th_min) * (Hr - 1)).long()

        idx_flat = (i * Wr + j).view(B, -1)  # (B,Hb*Wb)
        bev_flat = bev_feat.view(B, C, -1)  # (B,C,Hb*Wb)

        # 4. scatter-max → RV 피처
        rv_feat = torch.full((B, C, Hr * Wr), -1e9, device=dev)
        rv_feat.scatter_reduce_(2, idx_flat.unsqueeze(1).expand(-1, C, -1), bev_flat, reduce="amax")
        return rv_feat.view(B, C, Hr, Wr)

    # ───────────────── RV ➜ BEV (수학식 + scatter_max) ─────────────────────
    @staticmethod
    def sph_2_des_direct(rv_feat, r_map):
        """
        rv_feat : (B, C, Hr, Wr)   r_map : (B, 1, Hr, Wr)
        반환      : (B, C, Hb, Wb)  (Hb = Hr*8, Wb = Wr//4)
        """
        B, C, Hr, Wr = rv_feat.shape
        Hb, Wb = Hr * 8, Wr // 4
        dev = rv_feat.device

        # 1. θ, φ 실값 (라디안)
        phi_min, phi_max = map(math.radians, PHI_RANGE)
        th_min, th_max = map(math.radians, THETA_RANGE)

        j = torch.arange(Wr, device=dev).view(1, 1, 1, Wr)
        i = torch.arange(Hr, device=dev).view(1, 1, Hr, 1)
        phi = phi_min + j / (Wr - 1) * (phi_max - phi_min)
        theta = th_max - i / (Hr - 1) * (th_max - th_min)
        r = r_map  # (B,1,Hr,Wr)

        # 2. (x,y) 물리좌표
        x = r * torch.cos(theta) * torch.cos(phi)
        y = r * torch.cos(theta) * torch.sin(phi)

        # 3. BEV 인덱스
        u = ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (Wb - 1)).long()
        v = ((Y_RANGE[1] - y) / (Y_RANGE[1] - Y_RANGE[0]) * (Hb - 1)).long()
        idx_flat = (v * Wb + u).view(B, -1)  # (B,Hr*Wr)
        rv_flat = rv_feat.view(B, C, -1)  # (B,C,Hr*Wr)

        # 4. scatter-max → BEV 피처
        bev_feat = torch.full((B, C, Hb * Wb), -1e9, device=dev)
        bev_feat.scatter_reduce_(2, idx_flat.unsqueeze(1).expand(-1, C, -1), rv_flat, reduce="amax")
        return bev_feat.view(B, C, Hb, Wb)

    def des_2_sph(self, des, des_coord_curr, sph_coord_curr):
        BS, C, H, W = des.shape

        scale_rate, grid_to_point = descartes_scale_rates[H]
        point = grid_to_point(des, des_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=sph_coord_curr,
            output_size=(int(64 * scale_rate), int(2048 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def sph_2_des(self, sph, sph_coord_curr, des_coord_curr):
        BS, C, H, W = sph.shape

        scale_rate, grid_to_point = sphere_scale_rates[H]
        point = grid_to_point(sph, sph_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=des_coord_curr,
            output_size=(int(512 * scale_rate), int(512 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def forward(self, descartes_feat_in, z_map_256, r_map_32, descartes_coord_curr, sphere_coord_curr, deep_128_res):
        """
        descartes_feat_in : [BS, 192, 512, 512]
        z_map_256 : [BS, 1, 256, 256]
        r_map_32 : [BS, 1, 32, 1024]
        descartes_coord_curr : [BS, 160000, 2, 1]
        sphere_coord_curr : [BS, 160000, 2, 1]
        deep_128_res : [BS, 128, 64, 64]
        """
        z_map_128 = F.interpolate(z_map_256, size=(128, 128), mode="bilinear", align_corners=True)  # [BS, 1, 128, 128]
        r_map_16 = F.interpolate(r_map_32, size=(16, 512), mode="bilinear", align_corners=True)  # [BS, 1, 16, 512]

        # --------------- Descartes ---------------
        des0 = self.descartes_header(descartes_feat_in)  # [BS, 32, 256, 256]
        des0_as_sph = self.des_2_sph_direct(des0, z_map_256)  # [BS, 32, 32, 1024]
        sph0 = self.sphere_header(des0_as_sph)  # [BS, 32, 32, 1024]
        sph0_as_des = self.sph_2_des_direct(sph0, r_map_32)  # [BS, 32, 256, 256]
        des0 = torch.cat((des0, sph0_as_des), dim=1)  # [BS, 64, 256, 256]
        des0 = self.header_channel_down(des0)  # [BS, 32, 256, 256]

        des1 = self.descartes_res1(des0)  # [BS, 64, 128, 128]
        des1_as_sph = self.des_2_sph_direct(des1, z_map_128)  # [BS, 64, 128, 128]
        sph1 = self.sphere_res1(des1_as_sph)  # [BS, 64, 16, 512]
        sph1_as_des = self.sph_2_des_direct(sph1, r_map_16)  # [BS, 64, 128, 128]
        des1 = torch.cat((des1, sph1_as_des), dim=1)  # [BS, 128, 128, 128]
        des1 = self.res1_channel_down(des1)  # [BS, 64, 128, 128]

        des2 = self.descartes_res2(des1)  # [BS, 128, 64, 64]

        # --------------- Temporal Memory ---------------
        if deep_128_res is not None:
            fused = des2 + deep_128_res
            des2 = self.add_fuse(fused)

        # --------------- Decoder ---------------
        res0 = F.interpolate(des0, size=des0.size()[2:], mode="bilinear", align_corners=True)
        res1 = F.interpolate(des1, size=des0.size()[2:], mode="bilinear", align_corners=True)
        res2 = F.interpolate(des2, size=des0.size()[2:], mode="bilinear", align_corners=True)
        des_out = self.out_conv2(self.out_conv1(torch.cat([res0, res1, res2], dim=1)))  # [BS, 64, 32, 1024]

        # --------------- BackProject ---------------
        des_out_as_point = grid_2_point_scale_05(des_out, descartes_coord_curr)  # (BS, 64, 160000, 1)
        sph1_as_point = grid_2_point_scale_025(sph1, sphere_coord_curr)  # (BS, 64, 160000, 1)

        # --------------- Aux Head ---------------
        res0 = self.aux_head1(res0)
        res1 = self.aux_head2(res1)
        res2 = self.aux_head3(res2)

        return (
            des_out_as_point,  # (BS, 64, 160000, 1)
            sph1_as_point,  # (BS, 64, 160000, 1)
            res0,  # (BS, 3, 256, 256)
            res1,  # (BS, 3, 256, 256)
            res2,  # (BS, 3, 256, 256)
            des2,  # (BS, 128, 64, 64)
        )
