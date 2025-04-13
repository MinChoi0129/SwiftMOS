import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.polar_cartesian import Cart2Polar, Polar2Cart
from utils.pretty_print import shprint
from . import backbone


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
        # bilinear upsample 사용
        x_high_up = F.interpolate(x_high, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
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
    def __init__(self, base_block, context_layers, layers, nclasses=3, use_att=True):
        super(BEVNet, self).__init__()

        fusion_channels2 = context_layers[3] + context_layers[2]  # 128 + 64 = 192
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]  # 96 + 32 = 128
        self.out_channels = fusion_channels1 // 2  # 64

        # ----- Cartesian branch -----
        self.cart_header = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[0],
            context_layers[1],
            layers[0],
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.cart_res1 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[1],
            context_layers[2],
            layers[1],
            stride=2,
            dilation=1,
            use_att=use_att,
        )
        self.cart_res2 = self._make_layer(
            eval("backbone.{}".format(base_block)),
            context_layers[2] * 2,
            context_layers[3],
            layers[2],
            stride=2,
            dilation=1,
            use_att=use_att,
        )

        self.cart_up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        self.cart_up1 = AttMerge(context_layers[1], fusion_channels2, fusion_channels1 // 2, scale_factor=2)

        # ----- Polar branch -----
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
            context_layers[2] * 2,
            context_layers[3],
            layers[2],
            stride=2,
            dilation=1,
            use_att=use_att,
        )

        self.polar_up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        self.polar_up1 = AttMerge(context_layers[1], fusion_channels2, fusion_channels1 // 2, scale_factor=2)

        # ----- Convert Module -----
        self.p2c_12 = Polar2Cart((128, 128), (128, 128))
        self.c2p_12 = Cart2Polar((128, 128), (128, 128))
        self.p2c_3 = Polar2Cart((256, 256), (256, 256))

        self.conv_1 = backbone.BasicConv2d(448, 128, kernel_size=3, padding=1)
        self.conv_2 = backbone.BasicConv2d(128, self.out_channels, kernel_size=3, padding=1)
        self.aux_head1 = nn.Conv2d(128, nclasses, 1)
        self.aux_head2 = nn.Conv2d(192, nclasses, 1)
        self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, cart_input, polar_input):
        """
        입력:
          - cart_input: Cartesian branch 입력 (BS, Channels, H, W)
          - polar_input: Polar branch 입력 (BS, Channels, H, W)
          - coord_cart: 외부에서 전달받은 Cartesian 좌표 (BilinearSample 형식에 맞는 shape)
          - coord_polar: 외부에서 전달받은 Polar 좌표 (동일)
        출력:
          - fusion된 Cartesian BEV feature (BS, out_channels, H_final, W_final)
        """
        is_on_debug = False

        if is_on_debug:
            shprint(0, cart_input, polar_input)

        # Encoder 단계
        cart_in1 = self.cart_header(cart_input)  # 예: (BS, 32, 256, 256)
        polar_in1 = self.polar_header(polar_input)  # (BS, 32, 256, 256)

        cart_out1 = self.cart_res1(cart_in1)  # (BS, 64, 128, 128)
        polar_out1 = self.polar_res1(polar_in1)  # (BS, 64, 128, 128)

        if is_on_debug:
            shprint(1, cart_in1, polar_in1, cart_out1, polar_out1)

        # Bilinear sampling을 통한 교차 변환
        cart_out1_as_polar = self.c2p_12(cart_out1, polar_out1)  # (BS, 64, 128, 128)
        polar_out1_as_cart = self.p2c_12(polar_out1, cart_out1)  # (BS, 64, 128, 128)

        # 두 branch의 특징 concat
        polar_in2 = torch.cat((cart_out1_as_polar, polar_out1), dim=1)  # (BS, 128, 128, 128)
        cart_in2 = torch.cat((polar_out1_as_cart, cart_out1), dim=1)  # (BS, 128, 128, 128)

        if is_on_debug:
            shprint(2, cart_out1_as_polar, polar_out1_as_cart, polar_in2, cart_in2)

        cart_out2 = self.cart_res2(cart_in2)  # (BS, 128, 64, 64)
        polar_out2 = self.polar_res2(polar_in2)  # (BS, 128, 64, 64)

        if is_on_debug:
            shprint(3, cart_out2, polar_out2)

        # Decoder 단계
        cart_out3 = self.cart_up2(cart_out1, cart_out2)  # (BS, 96, 128, 128)
        polar_out3 = self.polar_up2(polar_out1, polar_out2)  # (BS, 96, 128, 128)

        cart_out3_as_polar = self.c2p_12(cart_out3, polar_out3)  # (BS, 96, 128, 128)
        polar_out3_as_cart = self.p2c_12(polar_out3, cart_out3)  # (BS, 96, 128, 128)

        if is_on_debug:
            shprint(4, cart_out3, polar_out3, cart_out3_as_polar, polar_out3_as_cart)

        polar_in3 = torch.cat((cart_out3_as_polar, polar_out3), dim=1)  # (BS, 192, 128, 128)
        cart_in3 = torch.cat((polar_out3_as_cart, cart_out3), dim=1)  # (BS, 192, 128, 128)

        if is_on_debug:
            shprint(5, polar_in3, cart_in3)

        cart_out4 = self.cart_up1(cart_in1, cart_in3)  # (BS, 64, 256, 256)
        polar_out4 = self.polar_up1(polar_in1, polar_in3)  # (BS, 64, 256, 256)

        polar_out4_as_cart = self.p2c_3(polar_out4, cart_out4)  # (BS, 64, 256, 256)

        final_cart = torch.cat((polar_out4_as_cart, polar_out4), dim=1)  # (BS, 128, 256, 256)

        if is_on_debug:
            shprint(6, cart_out4, polar_out4, polar_out4_as_cart, final_cart)

        res_0 = F.interpolate(cart_in2, size=cart_in1.size()[2:], mode="bilinear", align_corners=True)
        res_1 = F.interpolate(cart_in3, size=cart_in1.size()[2:], mode="bilinear", align_corners=True)
        res_2 = F.interpolate(final_cart, size=cart_in1.size()[2:], mode="bilinear", align_corners=True)
        res = [res_0, res_1, res_2]

        out = torch.cat(res, dim=1)

        if is_on_debug:
            shprint(res_0, res_1, res_2, out)
        out = self.conv_1(out)
        out = self.conv_2(out)

        res_0 = self.aux_head1(res_0)
        res_1 = self.aux_head2(res_1)
        res_2 = self.aux_head3(res_2)

        rtn = out, cart_out1, res_0, res_1, res_2
        if is_on_debug:
            shprint(*rtn)
        # [BS, 64, 256, 256], [BS, 64, 128, 128], [BS, 3, 256, 256], [BS, 3, 256, 256], [BS, 3, 256, 256]
        return rtn
