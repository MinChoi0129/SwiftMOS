import os
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import deep_point
from utils.pretty_print import shprint
from . import backbone
from networks.direct_view_converter import converters


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
        self.descartes_l1 = self._make_layer(backbone.BasicBlock, 192, 32, 1)
        self.sphere_l1 = self._make_layer(backbone.BasicBlock, 192, 32, 1)
        self.l1_channel_down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        # ----- ResBlock1 -----
        self.descartes_l2 = self._make_layer(backbone.BasicBlock, 32, 64, 2)
        self.sphere_l2 = self._make_layer(backbone.BasicBlock, 32, 64, 2)
        self.l2_channel_down = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # ----- ResBlock2 -----
        self.descartes_l3 = self._make_layer(backbone.BasicBlock, 64, 128, 3)

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

        try:
            single_batch = variable[0].cpu().numpy()
        except:
            single_batch = variable[0].detach().cpu().numpy()
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
            pcds_ind=sph_coord_curr[:, :, :2],
            output_size=(int(64 * scale_rate), int(2048 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def sph_2_des(self, sph, sph_coord_curr, des_coord_curr):
        BS, C, H, W = sph.shape

        scale_rate, grid_to_point = sphere_scale_rates[H]
        point = grid_to_point(sph, sph_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=des_coord_curr[:, :, :2],
            output_size=(int(512 * scale_rate), int(512 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def forward(self, descartes_feat_in, sphere_feat_in, descartes_coord_t_0, sphere_coord_t_0, deep_128_res):
        """
        descartes_feat_in : [BS, 192, 512, 512]
        descartes_coord_t_0 : [BS, 160000, 3, 1]
        sphere_coord_t_0 : [BS, 160000, 3, 1]
        deep_128_res : [BS, 128, 64, 64]
        """

        is_direct = True

        """Encoder"""
        des1 = self.descartes_l1(descartes_feat_in)  # (BS, C=32, H=256, W=256)
        sph1 = self.sphere_l1(sphere_feat_in)  # (BS, C=32, H=32, W=1024)

        if is_direct:
            sph1_as_des = self.sph_2_des_direct(sph1)  # (BS, C=32, H=256, W=256)
        else:
            sph1_as_des = self.sph_2_des(sph1, sphere_coord_t_0, descartes_coord_t_0)  # (BS, C=32, H=256, W=256)

        l1_concat = torch.cat((des1, sph1_as_des), dim=1)  # (BS, C=64, H=256, W=256)
        l2_des_in = self.l1_channel_down(l1_concat)  # (BS, C=32, H=256, W=256)

        des2 = self.descartes_l2(l2_des_in)  # (BS, C=64, H=128, W=128)
        sph2 = self.sphere_l2(sph1)  # (BS, C=64, H=16, W=512)

        if is_direct:
            sph2_as_des = self.sph_2_des_direct(sph2)  # (BS, C=64, H=128, W=128)
        else:
            sph2_as_des = self.sph_2_des(sph2, sphere_coord_t_0, descartes_coord_t_0)  # (BS, C=64, H=128, W=128)

        l2_concat = torch.cat((des2, sph2_as_des), dim=1)  # (BS, C=128, H=128, W=128)
        l3_des_in = self.l2_channel_down(l2_concat)  # (BS, C=64, H=128, W=128)

        des3 = self.descartes_l3(l3_des_in)  # (BS, C=128, H=64, W=64)

        """Temporal fusion"""
        if deep_128_res is not None:
            fused = des3 + deep_128_res  # (BS, C=128, H=64, W=64)
            des3 = self.add_fuse(fused)  # (BS, C=128, H=64, W=64)

        """Decoder"""
        out_size = des1.size()[2:]
        res1 = F.interpolate(des1, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=32, H=256, W=256)
        res2 = F.interpolate(des2, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=64, H=256, W=256)
        res3 = F.interpolate(des3, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=128, H=256, W=256)

        """Predictions"""
        des_out = self.out_conv2(self.out_conv1(torch.cat([res1, res2, res3], dim=1)))  # (BS, C=64, H=256, W=256)
        aux1 = self.aux_head1(res1)  # (BS, C=3, H=256, W=256)
        aux2 = self.aux_head2(res2)  # (BS, C=3, H=256, W=256)
        aux3 = self.aux_head3(res3)  # (BS, C=3, H=256, W=256)

        """Backprojection"""
        des_out_as_point = grid_2_point_scale_05(des_out, descartes_coord_t_0)  # (BS, C=64, N=160000, S=1)
        sph_out_as_point = grid_2_point_scale_05(sph1, sphere_coord_t_0)  # (BS, C=64, N=160000, S=1)

        return des_out_as_point, sph_out_as_point, aux1, aux2, aux3, des3
