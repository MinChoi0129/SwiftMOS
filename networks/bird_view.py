import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.polar_cartesian import Polar2Cart, Cart2Polar
from utils.pretty_print import shprint

from . import backbone

p2c_12 = Polar2Cart((128, 128), (128, 128))
c2p_12 = Cart2Polar((128, 128), (128, 128))
p2c_3 = Polar2Cart((256, 256), (256, 256))


class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.merge_layer = nn.Sequential(
            backbone.conv3x3(cin, cin // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cin // 2),
            backbone.act_layer,
            backbone.conv3x3(cin // 2, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer,
        )
        self.scale_factor = scale_factor

    def forward(self, x_low, x_high):
        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


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
        # pdb.set_trace()
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1)  # (BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # attention fusion
        ca_map = self.att_layer(x_merge.view(batch_size, 2 * self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)

        x_out = (x_merge * ca_map).sum(dim=1)  # (BS, channels, H, W)
        return x_out


class BEVNet(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(BEVNet, self).__init__()

        fusion_channels2 = context_layers[3] + context_layers[2]
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.out_channels = fusion_channels1 // 2

        """for Cartesian"""
        # encoder
        self.cart_header = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[0],  # 64
            context_layers[1],  # 32
            layers[0],  # 2
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.cart_res1 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[1],  # 32
            context_layers[2],  # 64
            layers[1],  # 3
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.cart_res2 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[2],  # 64
            context_layers[3],  # 128
            layers[2],  # 4
            stride=2,
            dilation=1,
            use_att=use_att,
        )

        # decoder
        self.cart_up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        self.cart_up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

        # 압축기
        self.conv_reduce_enc1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_enc2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_dec1 = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_dec2 = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.conv_reduce_com = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)

        """for Polar"""
        # encoder
        self.polar_header = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[0],
            context_layers[1],
            layers[0],
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.polar_res1 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[1],
            context_layers[2],
            layers[1],
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.polar_res2 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[2],
            context_layers[3],
            layers[2],
            stride=2,
            dilation=1,
            use_att=use_att,
        )

        # decoder
        self.polar_up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        self.polar_up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))

        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))

        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, f1, f2):  # x1: cart(BS, 192, 512, 512) / x2: polar(BS, 192, 512, 512)
        """
        x1: Cartesian input  (BS, 192, 512, 512)
        x2: Polar input      (BS, 192, 512, 512)
        return: cart_out4, polar_out4
        """

        # -----------------------------
        # (1) Encoder 단계
        # -----------------------------
        # Enc0
        cart_in1 = self.cart_header(f1)  # (BS, 32, 256, 256)
        polar_in1 = self.polar_header(f2)  # (BS, 32, 256, 256)

        # Enc1
        cart_out1 = self.cart_res1(cart_in1)  # (BS, 64, 128, 128)
        polar_out1 = self.polar_res1(polar_in1)  # (BS, 64, 128, 128)

        # ---- (Between Enc1–Enc2) 교차 변환 ----
        # cart_out1 -> polar_out1 (c2p_12), polar_out1 -> cart_out1 (p2c_12)
        cart_out1_as_polar = c2p_12(cart_out1, polar_out1)  # (BS, 64, 128, 128)
        polar_out1_as_cart = p2c_12(polar_out1, cart_out1)  # (BS, 64, 128, 128)

        # 교차 변환 결과와 기존 x1, x2를 concat 후 채널 축소
        polar_in2 = torch.cat((cart_out1_as_polar, polar_out1), dim=1)  # (BS, 128, 128, 128)
        polar_in2 = self.conv_reduce_enc1(polar_in2)  # (BS, 64, 128, 128)
        cart_in2 = torch.cat((polar_out1_as_cart, cart_out1), dim=1)  # (BS, 128, 128, 128)
        cart_in2 = self.conv_reduce_enc2(cart_in2)  # (BS, 64, 128, 128)

        # Enc2
        cart_out2 = self.cart_res2(cart_in2)  # (BS, 128, 64, 64)
        polar_out2 = self.polar_res2(polar_in2)  # (BS, 128, 64, 64)

        # -----------------------------
        # (2) Decoder 단계
        # -----------------------------
        # Dec1 (skip connection: Enc1 -> Dec1)
        cart_out3 = self.cart_up2(cart_out1, cart_out2)  # (BS, 96, 128, 128)
        polar_out3 = self.polar_up2(polar_out1, polar_out2)  # (BS, 96, 128, 128)

        # ---- (Between Dec1–Dec2) 교차 변환 ----
        cart_out3_as_polar = c2p_12(cart_out3, polar_out3)  # (BS, 96, 128, 128)
        polar_out3_as_cart = p2c_12(polar_out3, cart_out3)  # (BS, 96, 128, 128)

        polar_in3 = torch.cat((cart_out3_as_polar, polar_out3), dim=1)  # (BS, 192, 128, 128)
        polar_in3 = self.conv_reduce_dec2(polar_in3)  # (BS, 96, 128, 128)
        cart_in3 = torch.cat((polar_out3_as_cart, cart_out3), dim=1)  # (BS, 192, 128, 128)
        cart_in3 = self.conv_reduce_dec2(cart_in3)  # (BS, 96, 128, 128)

        # Dec2 (skip connection: Enc0 -> Dec2)
        cart_out4 = self.cart_up1(cart_in1, cart_in3)  # (BS, 64, 256, 256)
        polar_out4 = self.polar_up1(polar_in1, polar_in3)  # (BS, 64, 256, 256)

        polar_out4_as_cart = p2c_3(polar_out4, cart_out4)  # (BS, 128, 256, 256)

        final_cart = self.conv_reduce_com(torch.cat((polar_out4_as_cart, cart_out4), dim=1))  # (BS, 64, 256, 256)

        return final_cart  # (BS, 64, 256, 256)
