import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from networks import backbone, multi_view_network
from networks.backbone import CatFusion
import deep_point
from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax
import open3d as o3d
import yaml


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


class MOSNet(nn.Module):
    def __init__(self, pModel):
        super(MOSNet, self).__init__()
        self.pModel = pModel
        self.descartes_bev_shape = list(pModel.Voxel.descartes_shape)
        self.cylinder_bev_shape = list(pModel.Voxel.cylinder_shape)

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
        self.multi_view_network = multi_view_network.MultiViewNetwork()
        self.pred_layer = backbone.PredBranch(64, 3)
        self.point_post = CatFusion([64, 64, 64], 64)

    def stage_forward(
        self,
        xyzi,  # (BS, 3, 7, 160000, 1)
        descartes_coord,  # (BS, 3, 160000, 3, 1)
        cylinder_coord,  # (BS, 3, 160000, 2, 1)
        deep_128,  # (BS, 128, 64, 64)
    ):
        BS, T, C, N, _ = xyzi.shape

        # PointNet
        point_feats = self.point_pre(xyzi.view(BS * T, C, N, 1))  # (BS×3, 64, 160000, 1)

        # Descartes BEV 투영
        descartes_feat_in = VoxelMaxPool(
            pcds_feat=point_feats,
            pcds_ind=descartes_coord.view(BS * T, N, 3, 1)[:, :, :2],
            output_size=self.descartes_bev_shape[:2],
            scale_rate=(1.0, 1.0),
        ).view(BS, -1, *self.descartes_bev_shape[:2])

        # t_0 시점 데이터(피처, c좌표, p좌표)
        point_feats_t_0 = point_feats.view(BS, T, -1, N, 1)[:, 0].contiguous()  # (BS, 64, 160000, 1)
        descartes_coord_t_0 = descartes_coord[:, 0, :, :2].contiguous()
        cylinder_coord_t_0 = cylinder_coord[:, 0].contiguous()

        (
            des_out_as_point,  # (BS, 64, 160000, 1)
            cyl1_as_point,  # (BS, 64, 160000, 1)
            b0_256,  # (BS, 3, 256, 256)
            b1_256,  # (BS, 3, 256, 256)
            b2_256,  # (BS, 3, 256, 256)
            deep_128,  # (BS, 128, 64, 64)
        ) = self.multi_view_network(descartes_feat_in, descartes_coord_t_0, cylinder_coord_t_0, deep_128)

        point_feat_out = self.point_post(point_feats_t_0, des_out_as_point, cyl1_as_point)
        pred_cls = self.pred_layer(point_feat_out).float()  # (BS, 3, 160000, 1)

        return pred_cls, b0_256, b1_256, b2_256, deep_128

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

    def forward(
        self, xyzi_stages, descartes_coord_stages, cylinder_coord_stages, label_stages, descartes_label_stages
    ):
        """Start 3-Stage Forwarding"""
        stage = 3
        losses, loss_3d, loss_2d = [], [], []
        deep_128 = None
        for i in range(stage):
            xyzi_single = xyzi_stages[:, i].contiguous()
            descartes_coord_single = descartes_coord_stages[:, i].contiguous()
            cylinder_coord_single = cylinder_coord_stages[:, i].contiguous()

            pred_cls, b0_256, b1_256, b2_256, deep_128 = self.stage_forward(
                xyzi_single, descartes_coord_single, cylinder_coord_single, deep_128
            )

            bs, time_num, _, _ = pred_cls.shape
            b0_256 = b0_256.view(bs, time_num, -1).unsqueeze(-1)
            b1_256 = b1_256.view(bs, time_num, -1).unsqueeze(-1)
            b2_256 = b2_256.view(bs, time_num, -1).unsqueeze(-1)

            label_single = label_stages[:, i].contiguous()
            descartes_label_single = descartes_label_stages[:, i].contiguous().view(bs, -1, 1)

            l_3d = self._aux_loss(pred_cls, label_single, lovasz_scale=3)
            l_2d_0 = self._aux_loss(b0_256, descartes_label_single, lovasz_scale=3)
            l_2d_1 = self._aux_loss(b1_256, descartes_label_single, lovasz_scale=3)
            l_2d_2 = self._aux_loss(b2_256, descartes_label_single, lovasz_scale=3)
            l_2d = (l_2d_0 + l_2d_1 + l_2d_2) / 3

            loss = l_3d + l_2d
            losses.append(loss)
            loss_3d.append(l_3d)
            loss_2d.append(l_2d)

        loss = sum(losses) / stage
        loss_3d = sum(loss_3d) / stage
        loss_2d = sum(loss_2d) / stage

        return loss, loss_3d, loss_2d

    def infer(self, xyzi_single, descartes_coord_single, cylinder_coord_single, deep_128):
        pred_cls, b0_256, b1_256, b2_256, deep_128 = self.stage_forward(
            xyzi_single, descartes_coord_single, cylinder_coord_single, deep_128
        )
        return pred_cls, deep_128
