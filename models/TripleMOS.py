import os
import traceback
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import backbone, bird_view, range_view
from networks.backbone import BilinearSample, CatFusion, PointAttFusion, get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax
import open3d as o3d
import yaml
import copy

from utils.polar_cartesian import Cart2Polar, Polar2Cart
from utils.pretty_print import shprint


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(
        pcds_feat=pcds_feat.contiguous().float(),
        pcds_ind=pcds_ind.contiguous(),
        output_size=output_size,
        scale_rate=scale_rate,
    ).to(pcds_feat.dtype)
    return voxel_feat


grid_2_point_scale_full = backbone.BilinearSample(scale_rate=(1.0, 1.0))
grid_2_point_scale_05 = backbone.BilinearSample(scale_rate=(0.5, 0.5))


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel
        self.cart_bev_shape = list(pModel.Voxel.cart_bev_shape)
        self.polar_bev_shape = list(pModel.Voxel.polar_bev_shape)

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
        self.point_pre = backbone.PointNetStacker(7, 64, pre_bn=True, stack_num=2)
        self.pc_bev = bird_view.BEVNet()
        self.pred_layer = backbone.PredBranch(128, 3)

    def stage_forward(
        self,
        xyzi,  # BS, 3, 7, 160000, 1 (BS, 최근3프레임, 7=(x, y, z, intensity, dist, diff_x, diff_y), 160000(중 n개), 1)
        c_coord,  # BS, 3, 160000, 3, 1 (BS, 최근3프레임, 160000(중 n개), 3=(x,y,z quantized), 1)
        p_coord,  # BS, 3, 160000, 3, 1 (BS, 최근3프레임, 160000(중 n개), 3=(r,theta,z quantized), 1)
    ):
        BS, T, C, N, _ = xyzi.shape  # BS, 3, 7, 160000, 1

        """PointNet 기반 간단 처리"""
        point_feats = self.point_pre(xyzi.view(BS * T, C, N, 1))  # BSx3, 64, 160000, 1

        """입력 생성"""
        c_coord_t_0 = c_coord[:, 0, :, :2].contiguous()  # BS, 160000, 2(x_quan, y_quan), 1

        c_point_feats = VoxelMaxPool(  # BSx3, 64, 512, 512 (VoxelMaxPool) --> BS, 192, 512, 512 (View)
            pcds_feat=point_feats,
            pcds_ind=c_coord.view(BS * T, N, 3, 1)[:, :, :2],  # BSx3, 160000, 2, 1
            output_size=self.cart_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.cart_bev_shape[0], self.cart_bev_shape[1])

        p_point_feats = VoxelMaxPool(
            pcds_feat=point_feats,
            pcds_ind=p_coord.view(BS * T, N, 3, 1)[:, :, :2],
            output_size=self.polar_bev_shape[:2],  # (512, 512)
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, self.polar_bev_shape[0], self.polar_bev_shape[1])

        """모델 입력"""
        final_cart = self.pc_bev(c_point_feats, p_point_feats)  # [BS, 128, 256, 256]
        final_cart_as_point = grid_2_point_scale_05(final_cart, c_coord_t_0)  # [BS, 128, 160000, 1]

        """예측"""
        pred_cls = self.pred_layer(final_cart_as_point).float()  # [BS, 3, 160000, 1]
        return pred_cls

    @staticmethod
    def visualize_point_feature(pcds_xyzi, fused_point_feat, c=0):
        """
        현재 프레임의 point cloud를 open3d로 시각화 (특정 feature 채널로 색 입힘)
        단, 원점에서 거리 0~50인 점만 시각화하며, 배경은 검정색으로 설정

        Args:
            pcds_xyzi: Tensor [BS, 3, 7, N, 1]
            fused_point_feat: Tensor [BS, 64, N, 1]
            c: int - 사용할 feature 채널 인덱스 (0~63)
        """
        assert pcds_xyzi.shape[1] == 3, "입력은 최근 3프레임이어야 합니다."
        assert 0 <= c < fused_point_feat.shape[1], f"채널 c는 0~{fused_point_feat.shape[1]-1} 사이여야 함."

        # 현재 프레임 정보만 추출 (t=0)
        cur_pcd = pcds_xyzi[:, 0]  # [BS, 7, N, 1]
        xyz = cur_pcd[:, :3, :, 0]  # [BS, 3, N]
        feat = fused_point_feat[:, c, :, 0]  # [BS, N]

        # 배치 1개만 사용
        xyz_np = xyz[0].permute(1, 0).cpu().numpy()  # [N, 3]
        feat_np = feat[0].cpu().numpy()  # [N]

        # 🔹 원점에서의 거리 계산 후 필터링
        dist = np.linalg.norm(xyz_np, axis=1)
        mask = (dist >= 0) & (dist <= 50)

        xyz_np = xyz_np[mask]
        feat_np = feat_np[mask]

        # 색상 매핑
        norm_feat = (feat_np - feat_np.min()) / (feat_np.ptp() + 1e-8)
        colors = plt.cm.viridis(norm_feat)[:, :3]

        # Open3D 포인트 클라우드
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 🔧 Visualizer 사용해서 배경 검정색으로
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.array([0, 0, 0])  # 검정 배경
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    @staticmethod
    def save_feature_as_img(variable, variable_name, is_label=False):
        single_batch = variable[0].cpu().numpy()

        save_dir = f"/home/workspace/work/TripleMOS/images/{variable_name}"
        os.makedirs(save_dir, exist_ok=True)  # 🔧 폴더가 없으면 생성

        if is_label:
            plt.imsave(f"{save_dir}/sample.png", single_batch[:, :, 0])
        else:
            for i, c in enumerate(single_batch):
                plt.imsave(f"{save_dir}/{i:06}.png", c)

    def _aux_loss(self, pred, label, lovasz_scale):
        return self.criterion_seg_cate(pred, label) + lovasz_scale * lovasz_softmax(pred, label, ignore=0)

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        """
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        """
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def forward(self, xyzi, c_coord, p_coord, xyzi_raw, c_coord_raw, p_coord_raw, label):
        """Forward"""
        pred_cls = self.stage_forward(xyzi, c_coord, p_coord)
        pred_cls_raw = self.stage_forward(xyzi_raw, c_coord_raw, p_coord_raw)

        l_fused_3d = self._aux_loss(pred_cls, label, lovasz_scale=2)
        l_fused_3d_raw = self._aux_loss(pred_cls_raw, label, lovasz_scale=2)
        l_fused_3d_consistency = self.consistency_loss_l1(pred_cls, pred_cls_raw)

        loss = (l_fused_3d + l_fused_3d_raw) / 2 + l_fused_3d_consistency

        return loss, l_fused_3d, l_fused_3d_raw, l_fused_3d_consistency

    def infer(self, xyzi, c_coord, p_coord):
        pred_cls = self.stage_forward(xyzi, c_coord, p_coord)
        return pred_cls
