import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from utils.pretty_print import shprint

act_layer = nn.ReLU(inplace=True)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=bias,
    )


class DownSample2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DownSample2D, self).__init__()
        # self.in_planes, self.out_planes = in_planes, out_planes
        self.conv_branch = nn.Sequential(
            conv3x3(in_planes, out_planes, stride=stride, dilation=1),
            nn.BatchNorm2d(out_planes),
        )

        self.pool_branch = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1, dilation=1),
        )

        self.act = act_layer

    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_pool = self.pool_branch(x)
        x_out = self.act(x_conv + x_pool)
        return x_out


def get_module(param_dic, **kwargs):
    for key in param_dic:
        if (key != "type") and param_dic[key] is not None:
            kwargs[key] = param_dic[key]

    result_module = eval(param_dic["type"])(**kwargs)
    return result_module


class TConv(nn.Module):
    def __init__(self, T, cin, cout):
        super(TConv, self).__init__()
        self.T = T
        self.cin = cin
        self.cout = cout
        self.conv_t = nn.Sequential(
            nn.Conv2d(
                cin,
                cout,
                kernel_size=(3, 1),
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            act_layer,
        )

    def forward(self, x):
        BS_T, C, H, W = x.shape
        BS = BS_T // self.T

        x_out = self.conv_t(x.view(BS, self.T, C, H, W).transpose(1, 2).contiguous().view(BS, C, self.T, H * W))
        x_out = x_out.transpose(1, 2).contiguous().view(-1, self.cout, H, W)
        return x_out


class TConcat(nn.Module):
    def __init__(self, T, cin, cout):
        super(TConcat, self).__init__()
        self.T = T
        self.cin = cin
        self.cout = cout
        self.conv_tcat = nn.Sequential(
            nn.Conv2d(
                cin * T,
                cout,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            act_layer,
        )

    def forward(self, x):
        BS_T, C, H, W = x.shape
        BS = BS_T // self.T

        x_out = self.conv_tcat(x.view(BS, self.T * C, H, W))
        return x_out


class ChannelAtt(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAtt, self).__init__()
        self.cnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            act_layer,
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # channel wise
        ca_map = self.cnet(x)
        x = x * ca_map
        return x


class SpatialAtt(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SpatialAtt, self).__init__()
        self.snet = nn.Sequential(
            conv3x3(channels, 4, stride=1, dilation=1),
            nn.BatchNorm2d(4),
            act_layer,
            conv3x3(4, 1, stride=1, dilation=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # spatial wise
        sa_map = self.snet(x)
        x = x * sa_map
        return x


class CSAtt(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CSAtt, self).__init__()
        self.channel_att = ChannelAtt(channels, reduction)
        self.spatial_att = SpatialAtt(channels, reduction)

    def forward(self, x):
        # channel wise
        x1 = self.channel_att(x)
        x2 = self.spatial_att(x1)
        return x2


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        relu=True,
    ):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, reduction=1, dilation=1, use_att=True):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            conv3x3(
                in_planes=inplanes,
                out_planes=inplanes // reduction,
                stride=1,
                dilation=1,
            ),
            nn.BatchNorm2d(inplanes // reduction),
            act_layer,
            conv3x3(
                in_planes=inplanes // reduction,
                out_planes=inplanes,
                stride=1,
                dilation=dilation,
            ),
            nn.BatchNorm2d(inplanes),
        )

        self.use_att = use_att
        if self.use_att:
            self.channel_att = ChannelAtt(channels=inplanes, reduction=4)

        self.act = act_layer

    def forward(self, x):
        out = self.layer(x)
        if self.use_att:
            out = self.channel_att(out)

        out = self.act(out + x)
        return out


class BasicBlockv2(nn.Module):
    def __init__(self, inplanes, reduction=1, dilation=1, use_att=True):
        super(BasicBlockv2, self).__init__()
        self.layer = nn.Sequential(
            conv3x3(
                in_planes=inplanes,
                out_planes=inplanes // reduction,
                stride=1,
                dilation=1,
            ),
            nn.BatchNorm2d(inplanes // reduction),
            act_layer,
            conv3x3(
                in_planes=inplanes // reduction,
                out_planes=inplanes,
                stride=1,
                dilation=dilation,
            ),
            nn.BatchNorm2d(inplanes),
        )

        self.use_att = use_att
        if self.use_att:
            self.channel_att = CSAtt(channels=inplanes, reduction=4)

        self.act = act_layer

    def forward(self, x):
        out = self.layer(x)
        if self.use_att:
            out = self.channel_att(out)

        out = self.act(out + x)
        return out


class PredBranch(nn.Module):
    def __init__(self, cin, cout):
        super(PredBranch, self).__init__()
        self.pred_layer = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, dilation=1))

    def forward(self, x):
        x1 = F.dropout(x, p=0.2, training=self.training, inplace=False)
        pred = self.pred_layer(x1)
        return pred


class PointNet(nn.Module):
    def __init__(self, cin, cout, pre_bn=False, post_act=True):
        super(PointNet, self).__init__()
        self.layer = None
        if pre_bn and post_act:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                act_layer,
            )
        elif (not pre_bn) and post_act:
            self.layer = nn.Sequential(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                act_layer,
            )
        elif pre_bn and (not post_act):
            self.layer = nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
            )
        elif (not pre_bn) and (not post_act):
            self.layer = nn.Sequential(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        x_feat = self.layer(x)
        return x_feat


class PointNetStacker(nn.Module):
    def __init__(self, cin, cout, pre_bn=False, post_act=True, stack_num=1):
        super(PointNetStacker, self).__init__()
        layers = None
        if stack_num == 1:
            layers = [PointNet(cin=cin, cout=cout, pre_bn=pre_bn, post_act=post_act)]
        else:
            layers = [PointNet(cin=cin, cout=cout, pre_bn=pre_bn, post_act=True)]
            for i in range(1, stack_num - 1):
                layers.append(PointNet(cin=cout, cout=cout, pre_bn=False, post_act=True))

            layers.append(PointNet(cin=cout, cout=cout, pre_bn=False, post_act=post_act))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x_feat = self.layer(x)
        return x_feat


class BranchAttFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(BranchAttFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        self.weights = nn.Parameter(
            torch.ones(len(self.in_channel_list), dtype=torch.float32),
            requires_grad=True,
        )
        self.feat_model = nn.ModuleList()
        for i, in_channel in enumerate(self.in_channel_list):
            self.feat_model.append(PointNet(cin=in_channel, cout=out_channel, pre_bn=False))

    def forward(self, *x_list):
        weights = F.softmax(self.weights, dim=0)
        x_out = self.feat_model[0](x_list[0]) * weights[0]
        for i in range(1, len(x_list)):
            x_out = x_out + self.feat_model[i](x_list[i]) * weights[i]

        return x_out


class CatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(CatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel

        self.merge_layer = nn.Sequential(
            nn.Conv2d(s, s // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(s // 2),
            act_layer,
            nn.Conv2d(s // 2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            act_layer,
        )

    def forward(self, *x_list):
        x_merge = torch.cat(x_list, dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


class PointAttFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PointAttFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        self.att_layer = nn.Sequential(
            nn.Conv2d(
                len(self.in_channel_list) * out_channel,
                out_channel,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            act_layer,
            nn.Conv2d(out_channel, len(self.in_channel_list), kernel_size=1, padding=0),
        )

        # make feature layer
        self.feat_model = nn.ModuleList()
        for i, in_channel in enumerate(self.in_channel_list):
            self.feat_model.append(PointNet(cin=in_channel, cout=out_channel, pre_bn=False))

    def forward(self, *x_list):
        batch_size = x_list[0].shape[0]

        x_feat_list = [self.feat_model[i](x_list[i]) for i in range(len(x_list))]

        x_merge = torch.stack(x_feat_list, dim=1)  # (BS, S, channels, N, 1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        ca_map = self.att_layer(x_merge.view(batch_size, len(self.in_channel_list) * self.out_channel, -1, 1))
        ca_map = ca_map.view(batch_size, len(self.in_channel_list), 1, -1, 1)  # (BS, S, 1, N, 1)
        ca_map = F.softmax(ca_map, dim=1)  # (BS, S, 1, N, 1)

        x_out = (x_merge * ca_map).sum(dim=1)  # (BS, channels, N, 1)
        return x_out


class BilinearSample(nn.Module):
    def __init__(self, scale_rate):
        super(BilinearSample, self).__init__()
        self.scale_rate = scale_rate

    def forward(self, grid_feat, grid_coord):
        """
        Input:
            grid_feat, (BS, C, H, W)
            grid_coord, (BS, N, 2, S)
        Output:
            pc_feat, (BS, C, N, S)
        """
        H = grid_feat.shape[2]
        W = grid_feat.shape[3]

        grid_sample_x = (2 * grid_coord[:, :, 1] * self.scale_rate[1] / (W - 1)) - 1  # (BS, N, S)
        grid_sample_y = (2 * grid_coord[:, :, 0] * self.scale_rate[0] / (H - 1)) - 1  # (BS, N, S)

        grid_sample_2 = torch.stack((grid_sample_x, grid_sample_y), dim=-1)  # (BS, N, S, 2)
        pc_feat = F.grid_sample(
            grid_feat,
            grid_sample_2,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (BS, C, N, S)
        return pc_feat


# ---------- ViT(Self‑Attention) 모듈 ----------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=192, embed_dim=192, patch=32):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)  # [B, C_e, H/p, W/p]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, C_e]   (N=H/p*W/p)
        return x, H, W


class ViTEncoder(nn.Module):
    r"""
    Vision-Transformer encoder that preserves spatial size.

    Args
    ----
    in_ch     : 입력/출력 채널 수 (기본 192)
    patch     : 패치 크기 (16 ⇒ 32×32 토큰)
    depth     : Transformer layer 수
    nhead     : Multi-head 수  (= in_ch // 48 권장)
    mlp_ratio : FFN hidden 차수 비율
    drop_path : Stochastic depth 확률
    """

    def __init__(
        self,
        in_ch: int = 192,
        patch: int = 32,
        depth: int = 2,
        nhead: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(in_ch, in_ch, kernel_size=patch, stride=patch, bias=False)  # 512→32
        num_tokens = (512 // patch) ** 2

        self.pos = nn.Parameter(torch.zeros(1, num_tokens, in_ch))
        nn.init.trunc_normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_ch,
            nhead=nhead,
            dim_feedforward=int(in_ch * mlp_ratio),
            activation="gelu",
            batch_first=True,
        )
        dpr = torch.linspace(0, drop_path, depth)  # drop-path 구간별
        self.blocks = nn.ModuleList([_add_drop_path(encoder_layer, dpr[i].item()) for i in range(depth)])

        self.proj = nn.Conv2d(in_ch, in_ch, 1, bias=False)  # 채널 보정
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")

    # ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 192, 512, 512) → (B, 192, 512, 512)"""
        B, C, H, W = x.shape  # (B,192,512,512)
        assert H == W == 512, "지원 해상도는 512×512 고정입니다."

        # 1) 패치 임베딩
        tok = self.embed(x)  # (B,192,32,32)
        tok = tok.flatten(2).transpose(1, 2)  # (B,1024,192)

        # 2) 위치 부여 + Transformer Encoder
        tok = tok + self.pos
        for blk in self.blocks:
            tok = blk(tok)  # (B,1024,192)

        # 3) 원래 해상도로 복원
        feat = tok.transpose(1, 2).contiguous().reshape(B, C, 32, 32)
        feat = F.interpolate(feat, scale_factor=self.patch, mode="bilinear", align_corners=False)

        # 4) 1×1 conv + 입력 잔차
        out = self.proj(feat) + x  # (B,192,512,512)
        return out


# ----- util: stochastic depth 삽입 ------------------------------------------
def _add_drop_path(layer: nn.TransformerEncoderLayer, drop_prob: float):
    if drop_prob == 0.0:
        return layer
    from timm.models.layers import DropPath  # timm ≥0.9 필요

    layer.drop_path = DropPath(drop_prob)
    return layer
