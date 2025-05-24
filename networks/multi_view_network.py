import os
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import deep_point
from . import backbone
from utils.rv_bev import converters


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


grid_2_point_scale_full = backbone.BilinearSample((1.0, 1.0))
grid_2_point_scale_05 = backbone.BilinearSample((0.5, 0.5))
grid_2_point_scale_025 = backbone.BilinearSample((0.25, 0.25))
grid_2_point_scale_0125 = backbone.BilinearSample((0.125, 0.125))

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
        os.makedirs(save_dir, exist_ok=True)

        single_batch = variable[0].cpu().numpy()
        for i, c in enumerate(single_batch):
            plt.imsave(f"{save_dir}/{i:06}.png", c)

    @staticmethod
    def des_2_sph_direct(bev_feat: torch.Tensor) -> torch.Tensor:
        """
        BEV → RV direct max‐pool
        bev_feat: (B, C, Hb, Wb)
        return  : (B, C, Hr, Wr), Hr=Hb//8, Wr=Wb*4
        """
        return converters["BEV2RV"][bev_feat.shape[2]](bev_feat)

    @staticmethod
    def sph_2_des_direct(rv_feat: torch.Tensor) -> torch.Tensor:
        """
        RV → BEV direct max‐pool
        rv_feat : (B, C, Hr, Wr)
        return  : (B, C, Hb, Wb), Hb=Hr*8, Wb=Wr//4
        """
        return converters["RV2BEV"][rv_feat.shape[2]](rv_feat)

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

    def forward(self, descartes_feat_in, descartes_coord_t_0, sphere_coord_t_0, deep_128_res):
        """
        descartes_feat_in : [BS, 192, 512, 512]
        descartes_coord_t_0 : [BS, 160000, 3, 1]
        sphere_coord_t_0 : [BS, 160000, 3, 1]
        deep_128_res : [BS, 128, 64, 64]
        """

        """ Encoder """

        # Descartes Encoder 0 -> BEV to RV
        des0 = self.descartes_header(descartes_feat_in)  # (BS, C=32, H=256, W=256)
        des0_as_sph = self.des_2_sph_direct(des0)  # (BS, C=32, H=32, W=1024)

        # Spherical Encoder 0 -> RV to BEV
        sph0 = self.sphere_header(des0_as_sph)  # (BS, C=32, H=32, W=1024)
        sph0_as_des = self.sph_2_des_direct(sph0)  # (BS, C=32, H=256, W=256)

        # Concatenate BEV and Spherical -> Channel Down
        des0 = torch.cat((des0, sph0_as_des), dim=1)  # (BS, C=64, H=256, W=256)
        des0 = self.header_channel_down(des0)  # (BS, C=32, H=256, W=256)

        # Descartes Encoder 1 -> BEV to RV
        des1 = self.descartes_res1(des0)  # (BS, C=64, H=128, W=128)
        des1_as_sph = self.des_2_sph_direct(des1)  # (BS, C=64, H=16, W=512)

        # Spherical Encoder 1 -> RV to BEV
        sph1 = self.sphere_res1(des1_as_sph)  # (BS, C=64, H=16, W=512)
        sph1_as_des = self.sph_2_des_direct(sph1)  # (BS, C=64, H=128, W=128)

        # Concatenate BEV and Spherical -> Channel Down
        des1 = torch.cat((des1, sph1_as_des), dim=1)  # (BS, C=128, H=128, W=128)
        des1 = self.res1_channel_down(des1)  # (BS, C=64, H=128, W=128)

        # Descartes Encoder 2
        des2 = self.descartes_res2(des1)  # (BS, C=128, H=64, W=64)

        # Temporal fusion
        if deep_128_res is not None:
            fused = des2 + deep_128_res  # (BS, C=128, H=64, W=64)
            des2 = self.add_fuse(fused)  # (BS, C=128, H=64, W=64)

        # Decoder
        out_size = des0.size()[2:]
        res0 = F.interpolate(des0, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=32, H=256, W=256)
        res1 = F.interpolate(des1, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=64, H=256, W=256)
        res2 = F.interpolate(des2, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=128, H=256, W=256)

        # Predictions (ChannelFused, ChannelSeparated)
        des_out = self.out_conv2(self.out_conv1(torch.cat([res0, res1, res2], dim=1)))  # (BS, C=64, H=256, W=256)
        res0 = self.aux_head1(res0)  # (BS, C=3, H=256, W=256)
        res1 = self.aux_head2(res1)  # (BS, C=3, H=256, W=256)
        res2 = self.aux_head3(res2)  # (BS, C=3, H=256, W=256)

        # 2-D → 3-D
        des_out_as_point = grid_2_point_scale_05(des_out, descartes_coord_t_0)  # (BS, C=64, N=160000, S=1)
        sph1_as_point = grid_2_point_scale_025(sph1, sphere_coord_t_0)  # (BS, C=64, N=160000, S=1)

        return des_out_as_point, sph1_as_point, res0, res1, res2, des2
