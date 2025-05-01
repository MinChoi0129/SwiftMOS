import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rep(t: torch.Tensor, B: int):
    """(N,2) → (B,N,1,2)  (batch 복제용)"""
    return t.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)


# ─────────────── Range-View ➜ BEV ────────────────
class RV2BEV(nn.Module):
    """
    rv_feat   : (B,C,Hr,Wr)           (row=θ_v, col=φ_h)
    ref_bev   : (B,C,Hb,Wb)           (x→열, y→행)   원점 = 중앙
    """

    def __init__(
        self,
        rv_size,  # (Hr,Wr)
        bev_size,  # (Hb,Wb)
        r_max,  # 최대 투영 반경 (m/pixel 단순화용)
        vert_row=None,  # RV에서 지평선에 해당하는 row (None ⇒ Hr//2)
        max_batch: int = 8,
    ):
        super().__init__()
        Hr, Wr = rv_size
        Hb, Wb = bev_size
        vert_row = Hr // 2 if vert_row is None else vert_row

        # --- BEV 픽셀 중심 좌표 (x,y) ---
        yy, xx = torch.meshgrid(torch.arange(Hb), torch.arange(Wb))
        y = (yy - Hb / 2 + 0.5) * r_max / (Hb / 2 - 0.5)  # m 단위
        x = (xx - Wb / 2 + 0.5) * r_max / (Wb / 2 - 0.5)

        # --- (x,y) → (φ, row) ---
        phi = (torch.atan2(y, x) + 2 * np.pi) % (2 * np.pi)  # 0~2π
        col = (Wr - 1) - phi / (2 * np.pi) * (Wr - 1)  # col=0 왼쪽
        row = torch.full_like(col, vert_row, dtype=torch.float32)  # 고정 θ_v

        grid = torch.stack((col, row), -1).view(-1, 2)
        grid[..., 0] = grid[..., 0] / Wr * 2 - 1  # 정규화
        grid[..., 1] = grid[..., 1] / Hr * 2 - 1
        self.register_buffer("grid", _rep(grid, max_batch), False)

        yy_mask, xx_mask = torch.meshgrid(torch.arange(Hb), torch.arange(Wb))
        dst_xy = torch.stack((yy_mask, xx_mask), -1).view(-1, 2)
        bid = torch.arange(max_batch).unsqueeze(1).repeat(1, dst_xy.size(0))
        self.register_buffer(
            "oidx",
            torch.cat((bid.unsqueeze(-1), dst_xy.repeat(max_batch, 1, 1)), -1)
            .view(-1, 3)
            .long(),
            False,
        )

    def forward(self, rv_feat: torch.Tensor, ref_bev: torch.Tensor):
        B, C, _, _ = ref_bev.shape
        grid = self.grid[:B].to(rv_feat.device)
        oidx = self.oidx[: B * grid.size(1)].to(rv_feat.device)

        bev = ref_bev.clone()
        sampled = (
            F.grid_sample(rv_feat, grid, align_corners=True)
            .permute(0, 2, 1, 3)  # (B,N,C,1)
            .reshape(-1, C)
        )
        bev[oidx[:, 0], :, oidx[:, 1], oidx[:, 2]] = sampled
        return bev


# ─────────────── BEV ➜ Range-View ────────────────
class BEV2RV(nn.Module):
    """
    bev_feat : (B,C,Hb,Wb)
    ref_rv   : (B,C,Hr,Wr)
    """

    def __init__(self, rv_size, bev_size, r_max, vert_row=None, max_batch: int = 8):
        super().__init__()
        Hr, Wr = rv_size
        Hb, Wb = bev_size
        vert_row = Hr // 2 if vert_row is None else vert_row

        # --- RV 픽셀 → (x,y) 평면 투영 (r 고정) ---
        row, col = torch.meshgrid(torch.arange(Hr), torch.arange(Wr))
        phi = (Wr - 1 - col) / (Wr - 1) * 2 * np.pi
        r = r_max  # 단일 반경 슬라이스
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)

        idx_x = (x / r_max * (Wb / 2 - 0.5)) + (Wb / 2 - 0.5)
        idx_y = (y / r_max * (Hb / 2 - 0.5)) + (Hb / 2 - 0.5)

        grid = torch.stack((idx_x, idx_y), -1).view(-1, 2)
        grid[..., 0] = grid[..., 0] / Wb * 2 - 1
        grid[..., 1] = grid[..., 1] / Hb * 2 - 1
        self.register_buffer("grid", _rep(grid, max_batch), False)

        rc = torch.stack((row, col), -1).view(-1, 2)
        bid = torch.arange(max_batch).unsqueeze(1).repeat(1, rc.size(0))
        self.register_buffer(
            "oidx",
            torch.cat((bid.unsqueeze(-1), rc.repeat(max_batch, 1, 1)), -1)
            .view(-1, 3)
            .long(),
            False,
        )

    def forward(self, bev_feat: torch.Tensor, ref_rv: torch.Tensor):
        B, C, _, _ = ref_rv.shape
        grid = self.grid[:B].to(bev_feat.device)
        oidx = self.oidx[: B * grid.size(1)].to(bev_feat.device)

        rv = ref_rv.clone()
        sampled = (
            F.grid_sample(bev_feat, grid, align_corners=True)
            .permute(0, 2, 1, 3)
            .reshape(-1, C)
        )
        rv[oidx[:, 0], :, oidx[:, 1], oidx[:, 2]] = sampled
        return rv
