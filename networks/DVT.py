import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter
from config.config_MOS import get_config
from torch_scatter import scatter_add

general_config, dataset_param_config, model_param_config, optimize_param_config = get_config()
range_theta = general_config.Voxel.range_theta
sphere_voxel_r = general_config.Voxel.sphere_shape[2]
range_phi = general_config.Voxel.range_phi


# ────────────────────────────────
# RV ➜ BEV (Sphere → Descartes)
# ────────────────────────────────
class RV2BEV(nn.Module):
    def __init__(
        self,
        rv_size=(64, 2048),  # (H_r, W_r)  ← sphere_shape[:2]
        bev_size=(512, 512),  # (H_b, W_b)  ← descartes_shape[:2]
        r_range=(2.0, 50.0),  # ← range_r
        r_bins=30,  # ← sphere_shape[2]
        fov_phi=(math.radians(-180), math.radians(180)),  # ← range_phi [rad]
        fov_theta=(math.radians(-25), math.radians(3)),  # ← range_theta [rad]
        range_xy=(-50.0, 50.0, -50.0, 50.0),  # range_x, range_y
    ):
        super().__init__()
        self.H_r, self.W_r = rv_size
        self.H_b, self.W_b = bev_size
        self.r_min, self.r_max = r_range
        self.r_bins = r_bins

        self.phi_min, self.phi_max = fov_phi
        self.theta_min, self.theta_max = fov_theta
        self.xmin, self.xmax, self.ymin, self.ymax = range_xy

        # θ·φ mesh
        theta = torch.linspace(self.theta_max, self.theta_min, self.H_r).view(1, self.H_r, 1)
        phi = torch.linspace(self.phi_min, self.phi_max, self.W_r).view(1, 1, self.W_r)
        self.register_buffer("theta", theta)
        self.register_buffer("phi", phi)

    @torch.no_grad()
    def forward(self, rv_feat, rv_range_bin):
        B, C, _, _ = rv_feat.shape
        device = rv_feat.device

        # de-quantize r
        dr = (self.r_max - self.r_min) / self.r_bins
        r = rv_range_bin.squeeze(1) * dr + (self.r_min + dr / 2)

        theta, phi = self.theta.to(device), self.phi.to(device)
        x = r * torch.cos(theta) * torch.cos(phi)
        y = r * torch.cos(theta) * torch.sin(phi)

        u = ((x - self.xmin) / (self.xmax - self.xmin) * (self.W_b - 1)).long()
        v = ((self.ymax - y) / (self.ymax - self.ymin) * (self.H_b - 1)).long()
        valid = (u >= 0) & (u < self.W_b) & (v >= 0) & (v < self.H_b)

        bev = torch.zeros(B, C, self.H_b, self.W_b, device=device)
        for b in range(B):
            m = valid[b]
            if not m.any():
                continue
            idx = (v[b][m] * self.W_b + u[b][m]).view(-1)
            src = rv_feat[b, :, m]
            flat = scatter(src, idx.expand(C, -1), 1, dim_size=self.H_b * self.W_b, reduce="max")
            bev[b] = flat.view(C, self.H_b, self.W_b)
        return bev


class BEV2RV(nn.Module):
    """
    LiDAR 센서 원점 기준 BEV → RV 변환
      • bev_feat : (B, C, 512, 512)
      • bev_z_bin: (B, 1, 512, 512) ─ Quantize+VoxelMaxPool 결과 (센서 z-bin)
      → rv_feat  : (B, C, 64, 2048)
    """

    def __init__(
        self,
        bev_size=(512, 512),  # (H_b, W_b)
        rv_size=(64, 2048),  # (H_r, W_r)
        z_range=(-4.0, 2.0),  # 센서 z 범위 [m]
        z_bins=30,
        fov_phi=(math.radians(-180), math.radians(180)),
        fov_theta=(math.radians(-25), math.radians(3)),
        range_xy=(-50.0, 50.0, -50.0, 50.0),  # x_min,x_max,y_min,y_max
    ):
        super().__init__()
        self.H_b, self.W_b = bev_size
        self.H_r, self.W_r = rv_size

        self.z_min, self.z_max = z_range
        self.z_bins = z_bins

        # φ, θ 범위 (단위: rad)
        self.phi_min, self.phi_max = fov_phi
        self.theta_min, self.theta_max = fov_theta

        self.xmin, self.xmax, self.ymin, self.ymax = range_xy

        # ───────────────────── BEV 평면 좌표 그리드 ─────────────────────
        #   행(row)  → y축(위→아래 : ymax → ymin)
        #   열(col)  → x축(왼→오  : xmin → xmax)
        y_lin = torch.linspace(self.ymax, self.ymin, self.H_b)  # (H_b,)
        x_lin = torch.linspace(self.xmin, self.xmax, self.W_b)  # (W_b,)

        yg, xg = torch.meshgrid(y_lin, x_lin)  # (H_b, W_b) each
        self.register_buffer("xg", xg)  # x 좌표
        self.register_buffer("yg", yg)  # y 좌표

        # ───────────────── BEV 픽셀마다 고정 φ 인덱스 미리 계산 ─────────────────
        phi = torch.atan2(yg, xg)  # (H_b, W_b)
        col_idx = (
            ((phi - self.phi_min) / (self.phi_max - self.phi_min) * (self.W_r - 1))  # 0‥W_r
            .round()
            .clamp(0, self.W_r - 1)
            .long()
        )
        self.register_buffer("col_idx", col_idx)  # 정수형 φ 인덱스

    @torch.no_grad()
    def forward(self, bev_feat: torch.Tensor, bev_z_bin: torch.Tensor) -> torch.Tensor:
        """
        bev_feat : (B, C, 512, 512)
        bev_z_bin: (B, 1, 512, 512) ─ z-bin index (0‥29)
        """
        B, C, _, _ = bev_feat.shape
        device = bev_feat.device

        # 1) z-bin  → 실제 z (센서 원점 기준)
        dz = (self.z_max - self.z_min) / self.z_bins  # bin 높이 [m]
        z_rel = bev_z_bin.squeeze(1).float() * dz + (self.z_min + dz / 2)  # (B, H_b, W_b)

        # 2) 고정 ρ, φ
        x, y = self.xg.to(device), self.yg.to(device)  # (H_b, W_b)
        rho = torch.sqrt(x**2 + y**2) + 1e-6  # (H_b, W_b)
        col = self.col_idx.to(device)  # (H_b, W_b)

        # 3) 출력 버퍼
        rv = torch.zeros(B, C, self.H_r, self.W_r, device=device)

        # 4) 배치별 처리 (JIT/CPU 대비 간단하게 루프 유지)
        for b in range(B):
            theta = torch.atan2(z_rel[b], rho)  # (H_b, W_b)  θ > 0
            # row: 위(25°) → 아래(3°)로 증가
            row = (
                ((self.theta_max - theta) / (self.theta_max - self.theta_min) * (self.H_r - 1))
                .round()
                .clamp(0, self.H_r - 1)
                .long()
            )

            valid = torch.isfinite(theta)  # NaN 방지
            if not valid.any():
                continue

            idx_flat = (row[valid] * self.W_r + col[valid]).view(-1)  # (N,)
            src = bev_feat[b, :, valid]  # (C, N)

            rv_flat = scatter(
                src, idx_flat.expand(C, -1), dim=1, dim_size=self.H_r * self.W_r, reduce="max"
            )  # or "mean"
            rv[b] = rv_flat.view(C, self.H_r, self.W_r)

        return rv


# ───────────────────────────────────────
# 해상도별 변환기 테이블
# ───────────────────────────────────────
sizes = {
    "BEV": {512: (512, 512), 256: (256, 256), 128: (128, 128)},
    "RV": {64: (64, 2048), 32: (32, 1024), 16: (16, 512)},
    "Cyl": {64: (64, 2048), 32: (32, 1024), 16: (16, 512)},
    "Pol": {64: (64, 2048), 32: (32, 1024), 16: (16, 512)},
}

converters = {
    ####### BEV & RV #######
    "BEV2RV": {
        512: BEV2RV(sizes["BEV"][512], sizes["RV"][64]),
        256: BEV2RV(sizes["BEV"][256], sizes["RV"][32]),
        128: BEV2RV(sizes["BEV"][128], sizes["RV"][16]),
    },
    "RV2BEV": {
        64: RV2BEV(sizes["RV"][64], sizes["BEV"][512]),
        32: RV2BEV(sizes["RV"][32], sizes["BEV"][256]),
        16: RV2BEV(sizes["RV"][16], sizes["BEV"][128]),
    },
}
#     ####### BEV & Cyl #######
#     "BEV2Cyl": {
#         512: BEV2Cyl(sizes["BEV"][512], sizes["Cyl"][64]),
#         256: BEV2Cyl(sizes["BEV"][256], sizes["Cyl"][32]),
#         128: BEV2Cyl(sizes["BEV"][128], sizes["Cyl"][16]),
#     },
#     "Cyl2BEV": {
#         64: Cyl2BEV(sizes["Cyl"][64], sizes["BEV"][512]),
#         32: Cyl2BEV(sizes["Cyl"][32], sizes["BEV"][256]),
#         16: Cyl2BEV(sizes["Cyl"][16], sizes["BEV"][128]),
#     },
#     ####### BEV & Pol #######
#     "BEV2Pol": {
#         512: BEV2Pol(sizes["BEV"][512], sizes["Pol"][64]),
#         256: BEV2Pol(sizes["BEV"][256], sizes["Pol"][32]),
#         128: BEV2Pol(sizes["BEV"][128], sizes["Pol"][16]),
#     },
#     "Pol2BEV": {
#         64: Pol2BEV(sizes["Pol"][64], sizes["BEV"][512]),
#         32: Pol2BEV(sizes["Pol"][32], sizes["BEV"][256]),
#         16: Pol2BEV(sizes["Pol"][16], sizes["BEV"][128]),
#     },
# }
