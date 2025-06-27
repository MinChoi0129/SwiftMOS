import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter
from config.config_MOS import get_config

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


# class BEV2RV(nn.Module):
#     """
#     LiDAR 센서 원점 기준 BEV → RV 변환
#       • bev_feat : (B, C, 512, 512)
#       • bev_z_bin: (B, 1, 512, 512) ─ Quantize+VoxelMaxPool 결과 (센서 z-bin)
#       → rv_feat  : (B, C, 64, 2048)
#     """

#     def __init__(
#         self,
#         bev_size=(512, 512),  # (H_b, W_b)
#         rv_size=(64, 2048),  # (H_r, W_r)
#         z_range=(-4.0, 2.0),  # 센서 z 범위 [m]
#         z_bins=30,
#         fov_phi=(math.radians(-180), math.radians(180)),
#         fov_theta=(math.radians(-25), math.radians(3)),
#         range_xy=(-50.0, 50.0, -50.0, 50.0),  # x_min,x_max,y_min,y_max
#     ):
#         super().__init__()
#         self.H_b, self.W_b = bev_size
#         self.H_r, self.W_r = rv_size

#         self.z_min, self.z_max = z_range
#         self.z_bins = z_bins

#         # φ, θ 범위 (단위: rad)
#         self.phi_min, self.phi_max = fov_phi
#         self.theta_min, self.theta_max = fov_theta

#         self.xmin, self.xmax, self.ymin, self.ymax = range_xy

#         # ───────────────────── BEV 평면 좌표 그리드 ─────────────────────
#         #   행(row)  → y축(위→아래 : ymax → ymin)
#         #   열(col)  → x축(왼→오  : xmin → xmax)
#         y_lin = torch.linspace(self.ymax, self.ymin, self.H_b)  # (H_b,)
#         x_lin = torch.linspace(self.xmin, self.xmax, self.W_b)  # (W_b,)

#         yg, xg = torch.meshgrid(y_lin, x_lin)  # (H_b, W_b) each
#         self.register_buffer("xg", xg)  # x 좌표
#         self.register_buffer("yg", yg)  # y 좌표

#         # ───────────────── BEV 픽셀마다 고정 φ 인덱스 미리 계산 ─────────────────
#         phi = torch.atan2(yg, xg)  # (H_b, W_b)
#         col_idx = (
#             ((phi - self.phi_min) / (self.phi_max - self.phi_min) * (self.W_r - 1))  # 0‥W_r
#             .round()
#             .clamp(0, self.W_r - 1)
#             .long()
#         )
#         self.register_buffer("col_idx", col_idx)  # 정수형 φ 인덱스

#     @torch.no_grad()
#     def forward(self, bev_feat: torch.Tensor, bev_z_bin: torch.Tensor) -> torch.Tensor:
#         """
#         bev_feat : (B, C, 512, 512)
#         bev_z_bin: (B, 1, 512, 512) ─ z-bin index (0‥29)
#         """
#         B, C, _, _ = bev_feat.shape
#         device = bev_feat.device

#         # 1) z-bin  → 실제 z (센서 원점 기준)
#         dz = (self.z_max - self.z_min) / self.z_bins  # bin 높이 [m]
#         z_rel = bev_z_bin.squeeze(1).float() * dz + (self.z_min + dz / 2)  # (B, H_b, W_b)

#         # 2) 고정 ρ, φ
#         x, y = self.xg.to(device), self.yg.to(device)  # (H_b, W_b)
#         rho = torch.sqrt(x**2 + y**2) + 1e-6  # (H_b, W_b)
#         col = self.col_idx.to(device)  # (H_b, W_b)

#         # 3) 출력 버퍼
#         rv = torch.zeros(B, C, self.H_r, self.W_r, device=device)

#         # 4) 배치별 처리 (JIT/CPU 대비 간단하게 루프 유지)
#         for b in range(B):
#             theta = torch.atan2(z_rel[b], rho)  # (H_b, W_b)  θ > 0
#             # row: 위(25°) → 아래(3°)로 증가
#             row = (
#                 ((self.theta_max - theta) / (self.theta_max - self.theta_min) * (self.H_r - 1))
#                 .round()
#                 .clamp(0, self.H_r - 1)
#                 .long()
#             )

#             valid = torch.isfinite(theta)  # NaN 방지
#             if not valid.any():
#                 continue

#             idx_flat = (row[valid] * self.W_r + col[valid]).view(-1)  # (N,)
#             src = bev_feat[b, :, valid]  # (C, N)

#             rv_flat = scatter(
#                 src, idx_flat.expand(C, -1), dim=1, dim_size=self.H_r * self.W_r, reduce="max"
#             )  # or "mean"
#             rv[b] = rv_flat.view(C, self.H_r, self.W_r)

#         return rv


class BEV2RV(nn.Module):
    """
    BEV feature  →  RV feature
      • 부동소수 / 정수 입력 모두 지원
      • z_low〜z_hint 구간을 동일 값으로 '수직 채우기'
      • φ 보간·복잡한 반복 제거 → 메모리·연산량 최소
    """

    def __init__(
        self,
        bev_size=(512, 512),
        rv_size=(64, 2048),
        z_range=(-4.0, 2.0),
        z_bins=30,
        z_low=-1.73,
        fov_phi=(math.radians(-180), math.radians(180)),
        fov_theta=(math.radians(-25), math.radians(3)),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()

        # ───────── basic params ─────────
        self.H_b, self.W_b = bev_size
        self.H_r, self.W_r = rv_size
        self.z_min, self.z_max = z_range
        self.z_bins = z_bins
        self.z_low = z_low
        self.phi_min, self.phi_max = fov_phi
        self.theta_min, self.theta_max = fov_theta
        self.xmin, self.xmax, self.ymin, self.ymax = range_xy

        # ───────── pre-compute BEV grids ─────────
        y_lin = torch.linspace(self.ymax, self.ymin, self.H_b)  # row(y) : top → bottom
        x_lin = torch.linspace(self.xmin, self.xmax, self.W_b)  # col(x) : left → right
        yg, xg = torch.meshgrid(y_lin, x_lin)  # 1.9.1 기본 'ij'

        self.register_buffer("xg", xg)  # (H_b,W_b)
        self.register_buffer("yg", yg)  # (H_b,W_b)

        rho = torch.sqrt(xg**2 + yg**2)  # (H_b,W_b)
        self.register_buffer("rho", rho.flatten())  # (N,)

        phi_center = torch.atan2(yg, xg)  # (H_b,W_b)
        self.register_buffer("phi_center", phi_center.flatten())  # (N,)

        # θ 값(행)  : z_low ↔ row_low
        z_low_tensor = torch.full_like(self.rho, z_low)  # (N,)  ← Tensor 로 변환
        theta_low = torch.atan2(z_low_tensor, self.rho)  # (N,)

        row_low = (self.theta_max - theta_low) / (self.theta_max - self.theta_min) * (self.H_r - 1)
        row_low = row_low.round().clamp(0, self.H_r - 1).long()
        self.register_buffer("row_low", row_low)  # (N,)

        # φ → col (한 칸만 사용)
        col_flat = (self.phi_center - self.phi_min) / (self.phi_max - self.phi_min) * (self.W_r - 1)
        col_flat = col_flat.round().clamp(0, self.W_r - 1).long()
        self.register_buffer("col_flat", col_flat)  # (N,)

    # ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        bev_feat: torch.Tensor,  # (B,C,H_b,W_b)
        bev_z_bin: torch.Tensor,  # (B,1,H_b,W_b)
        chunk: int = 8192,  # 픽셀 청크 크기
    ) -> torch.Tensor:
        B, C, _, _ = bev_feat.shape
        device, dtype = bev_feat.device, bev_feat.dtype

        # z-bin  →  z_hint
        dz = (self.z_max - self.z_min) / self.z_bins
        z_hint = bev_z_bin.squeeze(1).float() * dz + (self.z_min + dz / 2)  # (B,H_b,W_b)

        rho = self.rho.to(device)  # (N,)
        col_flat = self.col_flat.to(device)  # (N,)
        row_low = self.row_low.to(device)  # (N,)
        N = rho.numel()

        # dtype-별 초기값
        if torch.is_floating_point(bev_feat):
            init_val = -float("inf")
        else:
            init_val = torch.iinfo(dtype).min

        rv = torch.full((B, C, self.H_r, self.W_r), init_val, dtype=dtype, device=device)

        # ───────── batch loop ─────────
        for b in range(B):
            bev_flat = bev_feat[b].view(C, -1).contiguous()  # (C,N)
            z_max_flat = z_hint[b].flatten()  # (N,)

            # θ_high (각 BEV 픽셀)
            theta_high = torch.atan2(z_max_flat, rho)  # (N,)
            row_high = (self.theta_max - theta_high) / (self.theta_max - self.theta_min) * (self.H_r - 1)
            row_high = row_high.round().clamp(0, self.H_r - 1).long()  # (N,)

            row_start = torch.min(row_low, row_high)
            row_end = torch.max(row_low, row_high)

            # θ 행(0~H_r-1) 루프
            for r in range(self.H_r):
                mask = (row_start <= r) & (r <= row_end)
                if not mask.any():
                    continue

                idx_all = torch.nonzero(mask, as_tuple=False).squeeze(1)
                for s in range(0, idx_all.numel(), chunk):
                    sel = idx_all[s : s + chunk]

                    tgt_idx = col_flat[sel] + r * self.W_r  # (sel_len,)
                    src_val = bev_flat[:, sel]  # (C, sel_len)

                    rv[b].view(C, -1)[:] = scatter(
                        src_val, tgt_idx.unsqueeze(0).expand(C, -1), dim=1, out=rv[b].view(C, -1), reduce="max"
                    )

        # 초기값 → 0
        rv.masked_fill_(rv == init_val, 0)
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
