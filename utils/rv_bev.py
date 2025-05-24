import math
import torch
import torch.nn.functional as F
from torch import nn


# ───────────────────────────────────────
# BEV → RV  (2-D ▶ 2-D)
# ───────────────────────────────────────
class BEV2RV(nn.Module):
    def __init__(
        self,
        bev_size,  # (H_bev, W_bev)
        rv_size,  # (H_rv, W_rv)
        sensor_h=1.73,  # [m]
        max_range=50.0,  # [m]
        fov_h=(-math.pi, math.pi),
        fov_v=(-25 * math.pi / 180, 3 * math.pi / 180),
        range_xy=(-50.0, 50.0, -50.0, 50.0),  # xmin,xmax,ymin,ymax
    ):
        super().__init__()
        H_b, W_b = bev_size
        H_r, W_r = rv_size
        xmin, xmax, ymin, ymax = range_xy

        # ① RV 각도 격자
        phi = torch.linspace(fov_v[1], fov_v[0], H_r).view(-1, 1).repeat(1, W_r)
        theta = torch.linspace(fov_h[0], fov_h[1], W_r).view(1, -1).repeat(H_r, 1)

        # ② z=0 평면 교차점
        sin_phi = torch.sin(phi).clamp(min=-1.0, max=-0.017)
        r = (sensor_h / -sin_phi).clamp(max=max_range)
        x = r * torch.cos(phi) * torch.cos(theta)
        y = r * torch.cos(phi) * torch.sin(theta)

        # ③ BEV 픽셀 → 정규화(−1~1)
        x_pix = (x - xmin) / (xmax - xmin) * (W_b - 1)
        y_pix = (ymax - y) / (ymax - ymin) * (H_b - 1)
        grid_x = 2 * x_pix / (W_b - 1) - 1
        grid_y = 2 * y_pix / (H_b - 1) - 1

        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # (1,H_r,W_r,2)
        self.register_buffer("grid", grid)

    def forward(self, bev_feat):  # bev_feat (B,C,H_b,W_b)
        B = bev_feat.size(0)
        grid = self.grid.to(bev_feat.device)  # ★DDP-safe: 입력 device 사용
        return F.grid_sample(
            bev_feat,
            grid.repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


# ───────────────────────────────────────
# RV → BEV  (2-D ▶ 2-D)
# ───────────────────────────────────────
class RV2BEV(nn.Module):
    def __init__(
        self,
        rv_size,  # (H_rv, W_rv)
        bev_size,  # (H_bev, W_bev)
        sensor_h=1.73,
        max_range=50.0,
        fov_h=(-math.pi, math.pi),
        fov_v=(-25 * math.pi / 180, 3 * math.pi / 180),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()
        H_r, W_r = rv_size
        H_b, W_b = bev_size
        xmin, xmax, ymin, ymax = range_xy

        # ① BEV 픽셀 격자 → 실제 (x,y)
        ii, jj = torch.meshgrid(torch.arange(H_b), torch.arange(W_b))  # 기본 "ij"
        x = xmin + jj / (W_b - 1) * (xmax - xmin)
        y = ymax - ii / (H_b - 1) * (ymax - ymin)

        # ② (x,y) → 각도
        theta = torch.atan2(y, x)
        dist_xy = torch.sqrt(x**2 + y**2).clamp(min=1e-3)
        phi = torch.atan2(-torch.full_like(dist_xy, sensor_h), dist_xy)

        # ③ RV 픽셀
        col = (theta - fov_h[0]) / (fov_h[1] - fov_h[0]) * (W_r - 1)
        row = (fov_v[1] - phi) / (fov_v[1] - fov_v[0]) * (H_r - 1)

        # ④ FOV/거리 바깥 → 패딩 영역
        mask = (dist_xy > max_range) | (phi < fov_v[0]) | (phi > fov_v[1])
        col[mask], row[mask] = -2.0, -2.0

        # ⑤ 정규화
        grid_x = 2 * col / (W_r - 1) - 1
        grid_y = 2 * row / (H_r - 1) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # (1,H_b,W_b,2)
        self.register_buffer("grid", grid)

    def forward(self, rv_feat):  # rv_feat (B,C,H_r,W_r)
        B = rv_feat.size(0)
        grid = self.grid.to(rv_feat.device)  # ★DDP-safe
        return F.grid_sample(
            rv_feat,
            grid.repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


# ───────────────────────────────────────
# 해상도별 변환기 테이블
# ───────────────────────────────────────
sizes = {
    "BEV": {512: (512, 512), 256: (256, 256), 128: (128, 128)},
    "RV": {64: (64, 2048), 32: (32, 1024), 16: (16, 512)},
}

converters = {
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
