import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import deep_point
from networks.DVT import converters

from . import backbone


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
        self.descartes_l1 = self._make_layer(backbone.BasicBlock, 192, 32, 2)
        self.sphere_l1 = self._make_layer(backbone.BasicBlock, 32, 32, 1, stride=1)
        self.l1_channel_down = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        # ----- ResBlock1 -----
        self.descartes_l2 = self._make_layer(backbone.BasicBlock, 32, 64, 3)
        self.sphere_l2 = self._make_layer(backbone.BasicBlock, 64, 64, 2, stride=1)
        self.l2_channel_down = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # ----- ResBlock2 -----
        self.descartes_l3 = self._make_layer(backbone.BasicBlock, 64, 128, 4)

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
    def save_feature_as_img(variable, variable_name, channel_pool="mean"):
        save_dir = f"images/features"
        os.makedirs(save_dir, exist_ok=True)

        try:
            single_batch = variable[0].cpu().numpy()
        except:
            single_batch = variable[0].detach().cpu().numpy()

        if channel_pool == "mean":
            channel_mean = np.mean(single_batch, axis=0)
            plt.imsave(f"{save_dir}/{variable_name}.png", channel_mean)
        elif channel_pool == "max":
            channel_max = np.max(single_batch, axis=0)
            plt.imsave(f"{save_dir}/{variable_name}.png", channel_max)
        else:
            raise ValueError(f"Invalid channel_pool value: {channel_pool}")

    def transform_view(self, feat, des_coord, sph_coord, is_direct):
        if feat.shape[2] == feat.shape[3]:  # square(from : BEV)
            if is_direct:
                return self.des_2_sph_2d2d(feat, des_coord)
            else:
                return self.des_2_sph_2d3d2d(feat, des_coord, sph_coord)
        else:  # rectangular(from : RV)
            if is_direct:
                return self.sph_2_des_2d2d(feat, sph_coord)
            else:
                return self.sph_2_des_2d3d2d(feat, sph_coord, des_coord)

    def des_2_sph_2d2d(self, bev_feat, descartes_coord_t_0):
        BS, C, Hb, Wb = bev_feat.shape

        bev_z_in = VoxelMaxPool(
            pcds_feat=descartes_coord_t_0[:, :, 2:3, :].permute(0, 2, 1, 3).contiguous(),  # (BS, 1, N, 1)
            pcds_ind=descartes_coord_t_0[:, :, :2, :],  # (BS, N, 2, 1)
            output_size=(Hb, Wb),
            scale_rate=(Hb / 512, Wb / 512),
        ).view(BS, -1, Hb, Wb)

        return converters["BEV2RV"][Hb](bev_feat, bev_z_in), bev_z_in

    def sph_2_des_2d2d(self, rv_feat, sphere_coord_t_0):
        BS, C, Hr, Wr = rv_feat.shape

        sph_range_in = VoxelMaxPool(
            pcds_feat=sphere_coord_t_0[:, :, 2:3, :].permute(0, 2, 1, 3).contiguous(),  # (BS, 1, N, 1)
            pcds_ind=sphere_coord_t_0[:, :, :2, :],  # (BS, N, 2, 1)
            output_size=(Hr, Wr),
            scale_rate=(Hr / 64, Wr / 2048),
        ).view(BS, -1, Hr, Wr)

        return converters["RV2BEV"][Hr](rv_feat, sph_range_in), sph_range_in

    def des_2_sph_2d3d2d(self, des, des_coord_curr, sph_coord_curr):
        BS, C, H, W = des.shape

        scale_rate, grid_to_point = descartes_scale_rates[H]
        point = grid_to_point(des, des_coord_curr)

        return (
            VoxelMaxPool(
                pcds_feat=point,
                pcds_ind=sph_coord_curr[:, :, :2],
                output_size=(int(64 * scale_rate), int(2048 * scale_rate)),
                scale_rate=(scale_rate, scale_rate),
            ),
            None,
        )

    def sph_2_des_2d3d2d(self, sph, sph_coord_curr, des_coord_curr):
        BS, C, H, W = sph.shape

        scale_rate, grid_to_point = sphere_scale_rates[H]
        point = grid_to_point(sph, sph_coord_curr)

        return (
            VoxelMaxPool(
                pcds_feat=point,
                pcds_ind=des_coord_curr[:, :, :2],
                output_size=(int(512 * scale_rate), int(512 * scale_rate)),
                scale_rate=(scale_rate, scale_rate),
            ),
            None,
        )

    def forward(self, descartes_feat_in, des_coord_t0, sph_coord_t0, temporal_res):
        """
        descartes_feat_in : [BS, C=192, H, W]
        des_coord_t0 : [BS, N, 3, 1]
        sph_coord_t0 : [BS, N, 3, 1]
        temporal_res : [BS, C, H, W]
        """

        is_direct = True
        save_image = False

        """Encoder"""

        ## Layer-1 ##
        des1 = self.descartes_l1(descartes_feat_in)  # (BS, C=32, H=256, W=256)
        des1_as_sph, des1_bev_z_in = self.transform_view(des1, des_coord_t0, sph_coord_t0, is_direct)  # (BS, C=32, H=32, W=1024)
        sph1 = self.sphere_l1(des1_as_sph)  # (BS, C=32, H=32, W=1024)
        sph1_as_des, sph1_bev_z_in = self.transform_view(sph1, des_coord_t0, sph_coord_t0, is_direct)  # (BS, C=32, H=256, W=256)
        l1_concat = torch.cat((des1, sph1_as_des), dim=1)  # (BS, C=64, H=256, W=256)
        l1_fused = self.l1_channel_down(l1_concat)  # (BS, C=32, H=256, W=256)

        ## Layer-2 ##
        des2 = self.descartes_l2(l1_fused)  # (BS, C=64, H=128, W=128)
        des2_as_sph, des2_bev_z_in = self.transform_view(des2, des_coord_t0, sph_coord_t0, is_direct)  # (BS, C=64, H=16, W=512)
        sph2 = self.sphere_l2(des2_as_sph)  # (BS, C=64, H=16, W=512)
        sph2_as_des, sph2_bev_z_in = self.transform_view(sph2, des_coord_t0, sph_coord_t0, is_direct)  # (BS, C=64, H=128, W=128)
        l2_concat = torch.cat((des2, sph2_as_des), dim=1)  # (BS, C=128, H=128, W=128)
        l2_fused = self.l2_channel_down(l2_concat)  # (BS, C=64, H=128, W=128)

        # Layer-3 ##
        des3 = self.descartes_l3(l2_fused)  # (BS, C=128, H=64, W=64)

        """Temporal fusion"""
        if temporal_res is not None:
            fused = des3 + temporal_res  # (BS, C=64, H=128, W=128)
            des3 = self.add_fuse(fused)  # (BS, C=64, H=128, W=128)

        """Decoder"""
        out_size = des1.size()[2:]
        res1 = F.interpolate(l1_fused, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=32, H=256, W=256)
        res2 = F.interpolate(l2_fused, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=64, H=256, W=256)
        res3 = F.interpolate(des3, size=out_size, mode="bilinear", align_corners=True)  # (BS, C=128, H=256, W=256)

        """Predictions"""
        des_out = self.out_conv2(self.out_conv1(torch.cat([res1, res2, res3], dim=1)))  # (BS, C=64, H=256, W=256)
        aux1 = self.aux_head1(res1)  # (BS, C=3, H=256, W=256)
        aux2 = self.aux_head2(res2)  # (BS, C=3, H=256, W=256)
        aux3 = self.aux_head3(res3)  # (BS, C=3, H=256, W=256)

        if save_image:
            self.save_feature_as_img(des1, "1-des1")
            self.save_feature_as_img(des1_as_sph, "2-des1_as_sph")
            self.save_feature_as_img(sph1, "3-sph1")
            self.save_feature_as_img(sph1_as_des, "4-sph1_as_des")
            self.save_feature_as_img(l1_concat, "5-l1_concat")
            self.save_feature_as_img(l1_fused, "6-l1_fused")
            self.save_feature_as_img(des2, "7-des2")
            self.save_feature_as_img(des2_as_sph, "8-des2_as_sph")
            self.save_feature_as_img(sph2, "9-sph2")
            self.save_feature_as_img(sph2_as_des, "10-sph2_as_des")
            self.save_feature_as_img(l2_concat, "11-l2_concat")
            self.save_feature_as_img(l2_fused, "12-l2_fused")
            self.save_feature_as_img(des3, "13-des3")
            self.save_feature_as_img(des1_bev_z_in, "14-des1_bev_z_in")
            self.save_feature_as_img(sph1_bev_z_in, "15-sph1_bev_z_in")
            self.save_feature_as_img(des2_bev_z_in, "16-des2_bev_z_in")
            self.save_feature_as_img(sph2_bev_z_in, "17-sph2_bev_z_in")
            self.save_feature_as_img(des_out, "18-des_out")
            raise Exception("ALL_FEATURES_SAVED.")

        """Backprojection"""
        _, des_grid_to_point = descartes_scale_rates[des_out.shape[2]]
        _, sph_grid_to_point = sphere_scale_rates[sph1.shape[2]]

        des_out_as_point = des_grid_to_point(des_out, des_coord_t0)  # (BS, C=64, N=160000, S=1)
        sph_out_as_point = sph_grid_to_point(sph1, sph_coord_t0)  # (BS, C=32, N=160000, S=1)

        return des_out_as_point, sph_out_as_point, aux1, aux2, aux3, des3
