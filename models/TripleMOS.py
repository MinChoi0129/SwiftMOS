import traceback
import torch
import torch.nn as nn

from networks import backbone, bird_view, range_view
from networks.backbone import BilinearSample, CatFusion, get_module
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
        try:
            bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
            bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
            bev_base_block = self.pModel.BEVParam.base_block
            fusion_mode = self.pModel.fusion_mode
            point_feat_channels = bev_context_layer[0]
            bev_context_layer[0] = self.pModel.seq_num * bev_context_layer[0]

            # network
            self.point_pre = backbone.PointNetStacker(7, point_feat_channels, pre_bn=True, stack_num=2)
            self.pc_bev = bird_view.BEVNet(
                bev_base_block, bev_context_layer, bev_layers, self.pModel.Voxel, use_att=True
            )

            self.cartBEV_2_point_05 = BilinearSample(
                in_dim=self.pc_bev.out_channels,
                scale_rate=(0.5, 0.5),
            )

            point_fusion_channels = (point_feat_channels, self.pc_bev.out_channels, 64)
            self.point_post = eval("backbone.{}".format(fusion_mode))(
                in_channel_list=point_fusion_channels, out_channel=self.pModel.point_feat_out_channels
            )

            self.pred_layer = backbone.PredBranch(self.pModel.point_feat_out_channels, self.pModel.class_num)
        except:
            traceback.print_exc()

    def stage_forward(
        self,
        pcds_xyzi,  # BS, 3, 7, 160000, 1 (BS, 최근3프레임, 7=(x, y, z, intensity, dist, diff_x, diff_y), 160000(중 n개), 1)
        coord_cart,  # BS, 3, 160000, 3, 1 (BS, 최근3프레임, 160000(중 n개), 3=(x,y,z quantized), 1)
        coord_polar,  # BS, 3, 160000, 3, 1 (BS, 최근3프레임, 160000(중 n개), 3=(r,theta,z quantized), 1)
        is_training,
    ):
        BS, T, C, N, _ = pcds_xyzi.shape  # BS, 3, 7, 160000, 1

        """PointNet 기반 간단 처리"""
        point_feat_pre = self.point_pre(pcds_xyzi.view(BS * T, C, N, 1))  # BSx3, 64, 160000, 1
        point_feat_t_0 = point_feat_pre.view(BS, T, -1, N, 1)[:, 0].contiguous()  # BS, 64, 160000, 1

        """입력 생성"""
        cart_coord_t_0 = coord_cart[:, 0, :, :2].contiguous()  # BS, 160000, 2(x_quan, y_quan), 1
        cart_coord_t_1 = coord_cart[:, 1, :, :2].contiguous()  # BS, 160000, 2(x_quan, y_quan), 1
        cart_coord_t_2 = coord_cart[:, 2, :, :2].contiguous()  # BS, 160000, 2(x_quan, y_quan), 1

        polar_coord_t_0 = coord_polar[:, 0, :, :2].contiguous()  # BS, 160000, 2(r_quan, theta_quan), 1
        polar_coord_t_1 = coord_polar[:, 1, :, :2].contiguous()  # BS, 160000, 2(r_quan, theta_quan), 1
        polar_coord_t_2 = coord_polar[:, 2, :, :2].contiguous()  # BS, 160000, 2(r_quan, theta_quan), 1

        cart_input = VoxelMaxPool(  # BSx3, 64, 512, 512 (VoxelMaxPool) --> BS, 192, 512, 512 (View)
            pcds_feat=point_feat_pre,
            pcds_ind=coord_cart.view(BS * T, N, 3, 1)[:, :, :2],  # BSx3, 160000, 2, 1
            output_size=self.cart_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.cart_bev_shape[0], self.cart_bev_shape[1])

        polar_input = VoxelMaxPool(
            pcds_feat=point_feat_pre,
            pcds_ind=coord_polar.view(BS * T, N, 3, 1)[:, :, :2],
            output_size=self.polar_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.polar_bev_shape[0], self.polar_bev_shape[1])

        """모델 입력"""
        history = None
        if is_training:
            for cart_coord_t, polar_coord_t in [
                (cart_coord_t_2, polar_coord_t_2),
                (cart_coord_t_1, polar_coord_t_1),
                (cart_coord_t_0, polar_coord_t_0),
            ]:
                bev_feat_2d, point_feat_cart, res_0, res_1, res_2, history = self.pc_bev(
                    cart_input, polar_input, cart_coord_t, polar_coord_t, history
                )
        else:
            bev_feat_2d, point_feat_cart, res_0, res_1, res_2, history = self.pc_bev(
                cart_input, polar_input, cart_coord_t_0, polar_coord_t_0, history
            )

        bev_feat_3d = self.cartBEV_2_point_05(
            bev_feat_2d, cart_coord_t_0
        )  # [256, 512를 기준으로 quan한 좌표]-> 0.5(to left) 스케일링 필요

        """융합 및 예측"""
        fused_point_feat = self.point_post(
            point_feat_t_0,
            bev_feat_3d,
            point_feat_cart,
        )  # 모두 [BS, 64, 160000, 1]

        pred_cls = self.pred_layer(fused_point_feat).float()  # [BS, 3, 160000, 1]

        return pred_cls, res_0, res_1, res_2

    def forward(self, pcds_xyzi, pcds_coord, pcds_polar_coord, pcds_target, pcds_bev_target):
        """Forward"""
        pred_cls, res_0, res_1, res_2 = self.stage_forward(pcds_xyzi, pcds_coord, pcds_polar_coord, is_training=True)

        # shape 변환
        bs, time_num, _, _ = pred_cls.shape
        res_0 = res_0.view(bs, time_num, -1).unsqueeze(-1)
        res_1 = res_1.view(bs, time_num, -1).unsqueeze(-1)
        res_2 = res_2.view(bs, time_num, -1).unsqueeze(-1)
        pcds_bev_target = pcds_bev_target.view(bs, -1, 1)

        # loss 정의
        loss1 = self.criterion_seg_cate(pred_cls, pcds_target) + 3 * lovasz_softmax(pred_cls, pcds_target, ignore=0)
        loss2 = self.criterion_seg_cate(res_0, pcds_bev_target) + 3 * lovasz_softmax(res_0, pcds_bev_target, ignore=0)
        loss3 = self.criterion_seg_cate(res_1, pcds_bev_target) + 3 * lovasz_softmax(res_1, pcds_bev_target, ignore=0)
        loss4 = self.criterion_seg_cate(res_2, pcds_bev_target) + 3 * lovasz_softmax(res_2, pcds_bev_target, ignore=0)

        loss = loss1 + (loss2 + loss3 + loss4) / 3

        return loss

    def infer(self, pcds_xyzi, pcds_coord, pcds_polar_coord):
        pred_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_polar_coord, is_training=False)[0]
        return pred_cls
