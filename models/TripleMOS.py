import torch
import torch.nn as nn

from networks import backbone, bird_view, range_view
from networks.backbone import CatFusion, get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax

import yaml
import copy

from utils.pretty_print import shprint


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.contiguous().float(),
        pcds_ind=pcds_ind.contiguous(),
        output_size=output_size,
        scale_rate=scale_rate,
    ).to(pcds_feat.dtype)
    return voxel_feat


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.cart_bev_shape = list(pModel.Voxel.cart_bev_shape)
        self.polar_bev_shape = list(pModel.Voxel.polar_bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)

        self._build_network()
        self._build_loss()

    def generate_triple_inputs(self, pointNet_feat, coords, shape_scalars):
        coord_cart, coord_polar, coord_sphere = coords
        BS, T, C, N = shape_scalars

        ####################################################################################

        cart_input = VoxelMaxPool(  # BSx3, 64, 512, 512 (VoxelMaxPool) --> BS, 192, 512, 512 (View)
            pcds_feat=pointNet_feat,
            pcds_ind=coord_cart.view(BS * T, N, 3, 1)[:, :, :2],  # BSx3, 160000, 2, 1
            output_size=self.cart_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.cart_bev_shape[0], self.cart_bev_shape[1])

        polar_input = VoxelMaxPool(  # BSx3, 64, 512, 512 (VoxelMaxPool) --> BS, 192, 512, 512 (View)
            pcds_feat=pointNet_feat,
            pcds_ind=coord_polar.view(BS * T, N, 3, 1)[:, :, :2],  # BSx3, 160000, 2, 1
            output_size=self.polar_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.polar_bev_shape[0], self.polar_bev_shape[1])

        sphere_input = VoxelMaxPool(  # BS, 64, 64, 2048
            pcds_feat=pointNet_feat.view(BS, T, -1, N, 1)[:, 0],  # BS, 3, 64, 160000, 1
            pcds_ind=coord_sphere[:, 0],  # BS, 160000, 2, 1
            output_size=self.rv_shape,
            scale_rate=(1.0, 1.0),
        )

        return cart_input, polar_input, sphere_input

    def _build_loss(self):
        if self.pModel.loss_mode == "ce":
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == "ohem":
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)
        elif self.pModel.loss_mode == "wce":
            # 가중치 CE
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open("datasets/semantic-kitti.yaml", "r") as f:
                task_cfg = yaml.load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg["learning_map"][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0
            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def _build_network(self):
        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        self.point_pre = backbone.PointNetStacker(7, rv_context_layer[0], pre_bn=True, stack_num=2)

        ############################################## Bird Eye View ##############################################
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_context_layer[0] = self.pModel.seq_num * rv_context_layer[0]

        self.multi_bev_net = bird_view.BEVNet(
            self.pModel.BEVParam.base_block, bev_context_layer, bev_layers, use_att=True
        )
        self.bev_2_point = get_module(self.pModel.BEVParam.bev_grid2point, in_dim=self.multi_bev_net.out_channels)

        ############################################## Range View ##############################################
        rv_base_block = self.pModel.RVParam.base_block
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, use_att=True)
        self.rv_2_point = get_module(self.pModel.RVParam.rv_grid2point, in_dim=self.rv_net.out_channels)

        ############################################## Fusion ##############################################
        point_fusion_channels = (rv_context_layer[0], self.multi_bev_net.out_channels, self.rv_net.out_channels)
        self.fuse = eval('backbone.{}'.format(self.pModel.fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.pModel.point_feat_out_channels)
        self.pred_layer = backbone.PredBranch(self.pModel.point_feat_out_channels, self.pModel.class_num)
        self.pred_bev_layer = nn.Sequential(
            nn.Conv2d(rv_context_layer[0] + self.multi_bev_net.out_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            backbone.PredBranch(64, self.pModel.class_num)
        )


    def stage_forward(  # 160000(Train) / 130000(Val)
        self,
        point_feat,  # BS, 3, 7, 160000, 1
        coord_cart,  # BS, 3, 160000, 3, 1
        coord_sphere,  # BS, 3, 160000, 2, 1
        coord_polar,  # BS, 3, 160000, 3, 1
    ):
        BS, T, C, N, _ = point_feat.shape  # 1, 3, 7, 160000, 1

        # PointNet 기반 간단 처리
        point_feat_pre = self.point_pre(point_feat.view(BS * T, C, N, 1))  # BSx3, 64, 160000, 1

        # 입력 생성
        coords = [coord_cart, coord_polar, coord_sphere]
        shape_scalars = [BS, T, C, N]
        cart_input, polar_input, sphere_input = self.generate_triple_inputs(point_feat_pre, coords, shape_scalars)

        # 모델 입력
        bev_feat_2d = self.multi_bev_net(cart_input, polar_input)  # (BS, 64, 256, 256)
        rv_feat_2d = self.rv_net(sphere_input)  # (BS, 64, 64, 1024)

        # 역투영
        bev_feat_3d = self.bev_2_point(bev_feat_2d, coord_cart[:, 0, :, :2])
        rv_feat_3d = self.rv_2_point(rv_feat_2d, coord_sphere[:, 0])

        # 3차원 MLP 
        point_feat_current = point_feat_pre.view(BS, T, -1, N, 1)[:, 0]
        fused_point_feat = self.fuse(point_feat_current, bev_feat_3d, rv_feat_3d)
        fused_point_feat_bev = torch.cat((point_feat_current, bev_feat_3d), dim=1)

        pred_cls = self.pred_layer(fused_point_feat).float()
        pred_bev_cls = self.pred_bev_layer(fused_point_feat_bev).float()

        return pred_cls, pred_bev_cls

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_polar_coord, pcds_target):
        # BS, 3, 160000, 1 / pcds_target(label)은 BS, 160000, 1
        pred_cls, pred_bev_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_polar_coord)

        loss = self.criterion_seg_cate(pred_cls, pcds_target) + 2 * lovasz_softmax(pred_cls, pcds_target, ignore=0)
        loss_bev = self.criterion_seg_cate(pred_bev_cls, pcds_target) + 2 * lovasz_softmax(pred_bev_cls, pcds_target, ignore=0)

        loss = loss + loss_bev
        return loss

    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_polar_coord):
        pred_cls, _ = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_polar_coord)
        return pred_cls
