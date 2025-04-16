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


class BEVNet(nn.Module):
    def __init__(self, base_block, cntx_l, l, VOXEL, nclasses=3, use_att=True):
        super(BEVNet, self).__init__()

        sda = {"stride": 2, "dilation": 1, "use_att": use_att}
        fusion_channels2 = cntx_l[3] + cntx_l[2]  # 128 + 64 = 192
        fusion_channels1 = fusion_channels2 // 2 + cntx_l[1]  # 96 + 32 = 128

        # ----- Header -----
        self.cart_header = self._make_layer(eval("backbone.{}".format(base_block)), cntx_l[0], cntx_l[1], l[0], **sda)
        self.polar_header = self._make_layer(eval("backbone.{}".format(base_block)), 32, cntx_l[1], l[0], **sda)

        # ----- ResBlock -----
        self.cart_res1 = self._make_layer(
            eval("backbone.{}".format(base_block)), cntx_l[1] * 2, cntx_l[2], l[1], **sda
        )
        self.polar_res1 = self._make_layer(
            eval("backbone.{}".format(base_block)), cntx_l[1] * 2, cntx_l[2], l[1], **sda
        )
        self.cart_res2 = self._make_layer(
            eval("backbone.{}".format(base_block)), cntx_l[2] * 2, cntx_l[3] * 2, l[2], **sda
        )

        # ----- UpBlock -----
        self.cart_up2 = AttMerge(cntx_l[2] * 2, cntx_l[3] * 2, fusion_channels2 // 2, scale_factor=2)
        self.cart_up1 = AttMerge(cntx_l[1] * 2, fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

        self.out_channels = fusion_channels1 // 2  # 64

        # ----- Convert Module -----
        self.cartBEV_2_point_05 = backbone.BilinearSample(in_dim=4, scale_rate=(0.5, 0.5))
        self.polarBEV_2_point_05 = backbone.BilinearSample(in_dim=4, scale_rate=(0.5, 0.5))
        self.cartBEV_2_point_025 = backbone.BilinearSample(in_dim=4, scale_rate=(0.25, 0.25))
        self.polarBEV_2_point_025 = backbone.BilinearSample(in_dim=4, scale_rate=(0.25, 0.25))

        # self.conv_1 = backbone.BasicConv2d(256, 128, kernel_size=3, padding=1)
        # self.conv_2 = backbone.BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)
        # self.aux_head1 = nn.Conv2d(64, nclasses, 1)
        # self.aux_head2 = nn.Conv2d(128, nclasses, 1)
        # self.aux_head3 = nn.Conv2d(64, nclasses, 1)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, c, p, c_coord, p_coord, history=None):
        """
        c : [BS, 192, 512, 512]
        p : [BS, 192, 512, 512]
        c_coord : [BS, 160000, 2, 1]
        p_coord : [BS, 160000, 2, 1]
        """

        # ----- Encoder -----
        # Header 단계: Cartesian 및 Polar branch를 각각 처리 후 fusion

        c0 = self.cart_header(c)  # [BS, 32, 256, 256]
        c0_point = self.cartBEV_2_point_05(c0, c_coord)  # [BS, 32, 160000, 1]
        c0_polar = VoxelMaxPool(c0_point, p_coord, output_size=(512, 512), scale_rate=(1.0, 1.0))  # [BS, 32, 512, 512]
        c0_polar = self.polar_header(c0_polar)  # [BS, 32, 256, 256]
        c0_point = self.polarBEV_2_point_05(c0_polar, p_coord)
        c0_cart = VoxelMaxPool(  # 두번째 파라미터가 갖는 값 quan 기준 W/H * scale_rate = output_size. 예시 -> 512 기준 * ? = 256 -> ? = 0.5
            c0_point, c_coord, output_size=(256, 256), scale_rate=(0.5, 0.5)
        )  # [Bs, 32, 256, 256]
        c0 = torch.cat((c0, c0_cart), dim=1)  # [BS, 64, 256, 256]

        # ResBlock 단계
        c1 = self.cart_res1(c0)  # [BS, 64, 128, 128]
        c1_point_out = self.cartBEV_2_point_025(c1, c_coord)
        c1_polar = VoxelMaxPool(c1_point_out, p_coord, output_size=(256, 256), scale_rate=(0.5, 0.5))
        c1_polar = self.polar_res1(c1_polar)
        c1_point = self.polarBEV_2_point_025(c1_polar, p_coord)
        c1_cart = VoxelMaxPool(
            c1_point, c_coord, output_size=(128, 128), scale_rate=(0.25, 0.25)
        )  # [BS, 64, 128, 128]
        c1 = torch.cat((c1, c1_cart), dim=1)  # [BS, 128, 128, 128]

        c2 = self.cart_res2(c1)  # 여기서 c2는 가장 낮은 해상도의 feature map
        # ----- Decoder: UpBlock들을 사용하여 cross fusion -----
        c3 = self.cart_up2(c1, c2)
        c4 = self.cart_up1(c0, c3)

        # if history is not None:
        #     c4 += history

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
        # res_0 = F.interpolate(c0, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 64, 256, 256]
        # res_1 = F.interpolate(c1, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 128, 256, 256]
        # res_2 = F.interpolate(c4, size=c0.size()[2:], mode="bilinear", align_corners=True)  # [BS, 64, 256, 256]
        # res = [res_0, res_1, res_2]

        # out = torch.cat(res, dim=1)  # [BS, 256, 256, 256]

        # out = self.conv_1(out)
        # out = self.conv_2(out)

        # res_0 = self.aux_head1(res_0)  # [bs, 3, 256, 256]
        # res_1 = self.aux_head2(res_1)  # [bs, 3, 256, 256]
        # res_2 = self.aux_head3(res_2)  # [bs, 3, 256, 256]

        # [BS, 64, 256, 256], [BS, 64, 160000, 1], [BS, 3, 256, 256], [BS, 3, 256, 256], [BS, 3, 256, 256], [BS, 64, 256, 256]
        # return out, c1_point, res_0, res_1, res_2, c4

        return c4, c1_point_out
