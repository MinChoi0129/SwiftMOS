import math
import torch
import torch.nn.functional as F
from torch import nn


class BEV2RV(nn.Module):
    """Direct view transformation (BEV → RV) with θ = elevation, φ = azimuth."""

    def __init__(
        self,
        bev_size,  # (H_bev, W_bev)
        rv_size,  # (H_rv, W_rv)
        sensor_h=1.73,  # [m] sensor height above ground
        max_range=50.0,  # [m] max projection distance
        fov_phi=(-math.pi, math.pi),  # φ (azimuth) range
        fov_theta=(-25 * math.pi / 180, 3 * math.pi / 180),  # θ (elevation) range
        range_xy=(-50.0, 50.0, -50.0, 50.0),  # xmin,xmax,ymin,ymax (m)
    ):
        super().__init__()
        H_b, W_b = bev_size
        H_r, W_r = rv_size
        xmin, xmax, ymin, ymax = range_xy

        # ① RV 각도 격자 (행: θ, 열: φ)
        theta = torch.linspace(fov_theta[1], fov_theta[0], H_r).view(-1, 1).repeat(1, W_r)  # (H_r, W_r)
        phi = torch.linspace(fov_phi[0], fov_phi[1], W_r).view(1, -1).repeat(H_r, 1)  # (H_r, W_r)

        # ② z=0 평면 교차점 (x, y)
        sin_theta = torch.sin(theta).clamp(min=-1.0, max=-0.017)
        r = (sensor_h / -sin_theta).clamp(max=max_range)  # 거리 r
        x = r * torch.cos(theta) * torch.cos(phi)
        y = r * torch.cos(theta) * torch.sin(phi)

        # ③ BEV 픽셀 좌표 → (-1,1) 정규화
        x_pix = (x - xmin) / (xmax - xmin) * (W_b - 1)
        y_pix = (ymax - y) / (ymax - ymin) * (H_b - 1)
        grid_x = 2 * x_pix / (W_b - 1) - 1
        grid_y = 2 * y_pix / (H_b - 1) - 1

        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # (1, H_r, W_r, 2)
        self.register_buffer("grid", grid)  # 고정 버퍼 (DDP‑safe)

    def forward(self, bev_feat):  # bev_feat: (B, C, H_b, W_b)
        B = bev_feat.size(0)
        grid = self.grid.to(bev_feat.device)
        return F.grid_sample(
            bev_feat,
            grid.repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, H_r, W_r)


class RV2BEV(nn.Module):
    """Inverse view transformation (RV → BEV) with θ = elevation, φ = azimuth."""

    def __init__(
        self,
        rv_size,  # (H_rv, W_rv)
        bev_size,  # (H_bev, W_bev)
        sensor_h=1.73,
        max_range=50.0,
        fov_phi=(-math.pi, math.pi),
        fov_theta=(-25 * math.pi / 180, 3 * math.pi / 180),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()
        H_r, W_r = rv_size
        H_b, W_b = bev_size
        xmin, xmax, ymin, ymax = range_xy

        # ① BEV 픽셀 격자 → 실세계 (x, y)
        ii, jj = torch.meshgrid(
            torch.arange(H_b), torch.arange(W_b)
        )  # (H_b, W_b), default 'ij' indexing (PyTorch ≤1.9)
        x = xmin + jj / (W_b - 1) * (xmax - xmin)
        y = ymax - ii / (H_b - 1) * (ymax - ymin)

        # ② (x, y) → 각도 (φ: azimuth, θ: elevation)
        phi = torch.atan2(y, x)  # 수평 각
        dist_xy = torch.sqrt(x**2 + y**2).clamp(min=1e-3)
        theta = torch.atan2(-torch.full_like(dist_xy, sensor_h), dist_xy)  # 수직 각 (음수 = 아래)

        # ③ RV 픽셀 좌표
        col = (phi - fov_phi[0]) / (fov_phi[1] - fov_phi[0]) * (W_r - 1)
        row = (fov_theta[1] - theta) / (fov_theta[1] - fov_theta[0]) * (H_r - 1)

        # ④ 시야/거리 외 영역 → 패딩(-2)
        mask = (dist_xy > max_range) | (theta < fov_theta[0]) | (theta > fov_theta[1])
        col[mask], row[mask] = -2.0, -2.0

        # ⑤ (-1,1) 정규화
        grid_x = 2 * col / (W_r - 1) - 1
        grid_y = 2 * row / (H_r - 1) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # (1, H_b, W_b, 2)
        self.register_buffer("grid", grid)

    def forward(self, rv_feat):  # rv_feat: (B, C, H_r, W_r)
        B = rv_feat.size(0)
        grid = self.grid.to(rv_feat.device)
        return F.grid_sample(
            rv_feat,
            grid.repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, H_b, W_b)


class BEV2Pol(nn.Module):
    """Direct view transformation (BEV → Polar) with r∈[0,max_range], φ=azimuth."""

    def __init__(
        self,
        bev_size,
        pol_size,
        max_range=50.0,
        fov_phi=(-math.pi, math.pi),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()
        H_b, W_b = bev_size
        H_p, W_p = pol_size
        xmin, xmax, ymin, ymax = range_xy

        r_lin = torch.linspace(0.0, max_range, H_p).view(-1, 1).repeat(1, W_p)
        phi_lin = torch.linspace(fov_phi[0], fov_phi[1], W_p).view(1, -1).repeat(H_p, 1)

        x = r_lin * torch.cos(phi_lin)
        y = r_lin * torch.sin(phi_lin)

        x_pix = (x - xmin) / (xmax - xmin) * (W_b - 1)
        y_pix = (ymax - y) / (ymax - ymin) * (H_b - 1)
        grid_x = 2 * x_pix / (W_b - 1) - 1
        grid_y = 2 * y_pix / (H_b - 1) - 1

        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, bev_feat):
        B = bev_feat.size(0)
        return F.grid_sample(
            bev_feat,
            self.grid.to(bev_feat.device).repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class Pol2BEV(nn.Module):
    """Inverse view transformation (Polar → BEV) with r, φ."""

    def __init__(
        self,
        pol_size,
        bev_size,
        max_range=50.0,
        fov_phi=(-math.pi, math.pi),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()
        H_p, W_p = pol_size
        H_b, W_b = bev_size
        xmin, xmax, ymin, ymax = range_xy

        ii, jj = torch.meshgrid(torch.arange(H_b), torch.arange(W_b))
        x = xmin + jj / (W_b - 1) * (xmax - xmin)
        y = ymax - ii / (H_b - 1) * (ymax - ymin)

        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)

        row = r / max_range * (H_p - 1)
        col = (phi - fov_phi[0]) / (fov_phi[1] - fov_phi[0]) * (W_p - 1)

        mask = r > max_range
        col[mask], row[mask] = -2.0, -2.0

        grid_x = 2 * col / (W_p - 1) - 1
        grid_y = 2 * row / (H_p - 1) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, pol_feat):
        B = pol_feat.size(0)
        return F.grid_sample(
            pol_feat,
            self.grid.to(pol_feat.device).repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class BEV2Cyl(nn.Module):
    """
    Broadcast BEV features to every z-bin of the Cylinder map.
    Cylinder 좌표: (row=z-bin, col=φ)
    """

    def __init__(
        self,
        bev_size,
        cyl_size,
        sensor_h=1.73,
        max_range=50.0,
        fov_phi=(-math.pi, math.pi),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
    ):
        super().__init__()
        H_b, W_b = bev_size
        H_c, W_c = cyl_size
        xmin, xmax, ymin, ymax = range_xy

        # φ(azimuth) 벡터
        phi = torch.linspace(fov_phi[0], fov_phi[1], W_c)  # (W_c,)

        # ground-plane에서 최대 사거리 r=max_range 사용
        x_g = max_range * torch.cos(phi)  # (W_c,)
        y_g = max_range * torch.sin(phi)  # (W_c,)

        # BEV 픽셀 → (-1,1) 정규화
        x_pix = (x_g - xmin) / (xmax - xmin) * (W_b - 1)
        y_pix = (ymax - y_g) / (ymax - ymin) * (H_b - 1)
        grid_x = 2 * x_pix / (W_b - 1) - 1  # (W_c,)
        grid_y = 2 * y_pix / (H_b - 1) - 1  # (W_c,)

        # (H_c, W_c, 2) 로 broadcast
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (W_c,2)
        grid = grid.unsqueeze(0).repeat(H_c, 1, 1)  # (H_c,W_c,2)
        self.register_buffer("grid", grid.unsqueeze(0).float())  # (1,H_c,W_c,2)

    def forward(self, bev_feat):  # (B,C,H_b,W_b)
        B = bev_feat.size(0)
        return F.grid_sample(
            bev_feat,
            self.grid.to(bev_feat.device).repeat(B, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class Cyl2BEV(nn.Module):
    """
    Use the z-row corresponding to sensor height (z≈0) from Cylinder map
    and project it back to BEV.
    """

    def __init__(
        self,
        cyl_size,
        bev_size,
        sensor_h=1.73,
        fov_phi=(-math.pi, math.pi),
        range_xy=(-50.0, 50.0, -50.0, 50.0),
        z_range=None,  # (z_min,z_max); default = (-sensor_h,+sensor_h)
    ):
        super().__init__()
        H_c, W_c = cyl_size
        H_b, W_b = bev_size
        xmin, xmax, ymin, ymax = range_xy
        if z_range is None:
            z_range = (-sensor_h, sensor_h)
        z_min, z_max = z_range

        # z=0 (센서 높이) 가 위치하는 row 인덱스
        sensor_row = int(round((0.0 - z_min) / (z_max - z_min) * (H_c - 1)))
        sensor_row = max(0, min(H_c - 1, sensor_row))

        # (u,v) → (row,col) 계산
        ii, jj = torch.meshgrid(torch.arange(H_b), torch.arange(W_b))
        x = xmin + jj / (W_b - 1) * (xmax - xmin)
        y = ymax - ii / (H_b - 1) * (ymax - ymin)

        phi = torch.atan2(y, x)  # (H_b,W_b)
        col = (phi - fov_phi[0]) / (fov_phi[1] - fov_phi[0]) * (W_c - 1)
        row = torch.full_like(col, sensor_row, dtype=torch.float32)

        grid_x = 2 * col / (W_c - 1) - 1
        grid_y = 2 * row / (H_c - 1) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # (1,H_b,W_b,2)
        self.register_buffer("grid", grid)

    def forward(self, cyl_feat):  # (B,C,H_c,W_c)
        B = cyl_feat.size(0)
        return F.grid_sample(
            cyl_feat,
            self.grid.to(cyl_feat.device).repeat(B, 1, 1, 1),
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
    ####### BEV & Cyl #######
    "BEV2Cyl": {
        512: BEV2Cyl(sizes["BEV"][512], sizes["Cyl"][64]),
        256: BEV2Cyl(sizes["BEV"][256], sizes["Cyl"][32]),
        128: BEV2Cyl(sizes["BEV"][128], sizes["Cyl"][16]),
    },
    "Cyl2BEV": {
        64: Cyl2BEV(sizes["Cyl"][64], sizes["BEV"][512]),
        32: Cyl2BEV(sizes["Cyl"][32], sizes["BEV"][256]),
        16: Cyl2BEV(sizes["Cyl"][16], sizes["BEV"][128]),
    },
    ####### BEV & Pol #######
    "BEV2Pol": {
        512: BEV2Pol(sizes["BEV"][512], sizes["Pol"][64]),
        256: BEV2Pol(sizes["BEV"][256], sizes["Pol"][32]),
        128: BEV2Pol(sizes["BEV"][128], sizes["Pol"][16]),
    },
    "Pol2BEV": {
        64: Pol2BEV(sizes["Pol"][64], sizes["BEV"][512]),
        32: Pol2BEV(sizes["Pol"][32], sizes["BEV"][256]),
        16: Pol2BEV(sizes["Pol"][16], sizes["BEV"][128]),
    },
}
