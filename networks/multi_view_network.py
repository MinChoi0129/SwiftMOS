import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import deep_point
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

grid_2_point_scale_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))
grid_2_point_scale_05 = backbone.BilinearSample(scale_rate=(0.5, 0.5))
grid_2_point_scale_025 = backbone.BilinearSample(scale_rate=(0.25, 0.25))

descartes_scale_rates = {
    512: (1.0, grid_2_point_scale_full),
    256: (0.5, grid_2_point_scale_05),
    128: (0.25, grid_2_point_scale_025),
}
cylinder_scale_rates = {
    64: (1.0, grid_2_point_scale_full),
    32: (0.5, grid_2_point_scale_05),
    16: (0.25, grid_2_point_scale_025),
}


class MultiViewNetwork(nn.Module):
    def __init__(self):
        super(MultiViewNetwork, self).__init__()

        # ----- Header -----
        self.descartes_header = self._make_layer(backbone.BasicBlock, 192, 32, 2)
        self.cylinder_header = self._make_layer(backbone.BasicBlock, 32, 32, 1, stride=1)
        self.header_channel_down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        # ----- ResBlock1 -----
        self.descartes_res1 = self._make_layer(backbone.BasicBlock, 32, 64, 3)
        self.cylinder_res1 = self._make_layer(backbone.BasicBlock, 64, 64, 2, stride=1)
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
        save_dir = f"/home/workspace/work/TripleMOS/images/{variable_name}"
        os.makedirs(save_dir, exist_ok=True)  # 🔧 폴더가 없으면 생성

        single_batch = variable[0].cpu().numpy()
        for i, c in enumerate(single_batch):
            plt.imsave(f"{save_dir}/{i:06}.png", c)

    def des_2_cyl(self, des, des_coord_curr, cyl_coord_curr):
        BS, C, H, W = des.shape

        scale_rate, grid_to_point = descartes_scale_rates[H]
        point = grid_to_point(des, des_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=cyl_coord_curr,
            output_size=(int(64 * scale_rate), int(2048 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def cyl_2_des(self, cyl, cyl_coord_curr, des_coord_curr):
        BS, C, H, W = cyl.shape

        scale_rate, grid_to_point = cylinder_scale_rates[H]
        point = grid_to_point(cyl, cyl_coord_curr)

        return VoxelMaxPool(
            pcds_feat=point,
            pcds_ind=des_coord_curr,
            output_size=(int(512 * scale_rate), int(512 * scale_rate)),
            scale_rate=(scale_rate, scale_rate),
        )

    def forward(self, descartes_feat_in, descartes_coord_curr, cylinder_coord_curr, deep_128):
        """
        descartes_feat_in : [BS, 192, 512, 512]
        descartes_coord_curr : [BS, 160000, 2, 1]
        cylinder_coord_curr : [BS, 160000, 2, 1]
        deep_128 : [BS, 128, 64, 64]
        """

        # --------------- Descartes ---------------
        des0 = self.descartes_header(descartes_feat_in)  # [BS, 32, 256, 256]
        des0_to_cyl = self.des_2_cyl(des0, descartes_coord_curr, cylinder_coord_curr)  # [BS, 32, 32, 1024]
        cyl0 = self.cylinder_header(des0_to_cyl)  # [BS, 32, 32, 1024]
        cyl0_to_des = self.cyl_2_des(cyl0, cylinder_coord_curr, descartes_coord_curr)

        des0 = torch.cat((des0, cyl0_to_des), dim=1)
        des0 = self.header_channel_down(des0)  # [BS, 64, 256, 256]

        des1 = self.descartes_res1(des0)  # [BS, 64, 128, 128]
        des1_to_cyl = self.des_2_cyl(des1, descartes_coord_curr, cylinder_coord_curr)
        cyl1 = self.cylinder_res1(des1_to_cyl)  # [BS, 64, 16, 512]
        cyl1_to_des = self.cyl_2_des(cyl1, cylinder_coord_curr, descartes_coord_curr)
        des1 = torch.cat((des1, cyl1_to_des), dim=1)
        des1 = self.res1_channel_down(des1)

        des2 = self.descartes_res2(des1)  # [BS, 128, 64, 64]

        # --------------- Temporal Memory ---------------
        if deep_128 is not None:
            fused = des2 + deep_128  #
            des2 = self.add_fuse(fused)

        # --------------- Decoder ---------------
        res0 = F.interpolate(des0, size=des0.size()[2:], mode="bilinear", align_corners=True)
        res1 = F.interpolate(des1, size=des0.size()[2:], mode="bilinear", align_corners=True)
        res2 = F.interpolate(des2, size=des0.size()[2:], mode="bilinear", align_corners=True)
        des_out = self.out_conv2(self.out_conv1(torch.cat([res0, res1, res2], dim=1)))  # [BS, 64, 32, 1024]

        # --------------- BackProject ---------------
        des_out_as_point = grid_2_point_scale_05(des_out, descartes_coord_curr)
        cyl1_as_point = grid_2_point_scale_025(cyl1, cylinder_coord_curr)

        # --------------- Aux Head ---------------
        res0 = self.aux_head1(res0)
        res1 = self.aux_head2(res1)
        res2 = self.aux_head3(res2)

        return (
            des_out_as_point,  # (BS, 64, 160000, 1)
            cyl1_as_point,  # (BS, 64, 160000, 1)
            res0,  # (BS, 3, 256, 256)
            res1,  # (BS, 3, 256, 256)
            res2,  # (BS, 3, 256, 256)
            des2,  # (BS, 128, 64, 64)
        )
