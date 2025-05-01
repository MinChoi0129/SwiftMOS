# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Polar2Cart(nn.Module):
#     def __init__(
#         self, polar_size, cart_size, center_drop_grid_size=3.0, max_batch=8, mode="in"
#     ):
#         super(Polar2Cart, self).__init__()
#         if mode == "in":
#             yy_org, xx_org = torch.meshgrid(
#                 torch.arange(cart_size[0]), torch.arange(cart_size[1])
#             )
#             yy = yy_org - cart_size[0] / 2.0 + 0.5
#             xx = xx_org - cart_size[1] / 2.0 + 0.5

#             self.yy_org = yy_org
#             self.xx_org = xx_org
#             self.yy = yy
#             self.xx = xx

#             depth = torch.sqrt(xx**2 + yy**2)
#             phi = np.pi - torch.atan2(yy, xx)
#             index_y = (
#                 depth
#                 / (cart_size[0] / 2.0 * np.sqrt(2.0))
#                 * (polar_size[0] + center_drop_grid_size)
#                 - center_drop_grid_size
#             )
#             index_x = phi / np.pi / 2.0 * polar_size[1]

#             mask = (index_y > 0).view(-1)
#             index_y = index_y / polar_size[0] * 2.0 - 1.0
#             index_x = index_x / polar_size[1] * 2.0 - 1.0

#             grid_sample_index = torch.stack((index_x, -index_y), axis=-1).view(-1, 2)
#             grid_sample_index = grid_sample_index[mask, :].unsqueeze(0).unsqueeze(2)
#             self.grid_sample_index = nn.Parameter(
#                 grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False
#             )

#             grid_sample_xy = torch.stack((yy_org, xx_org), axis=-1).view(-1, 2)
#             grid_sample_xy = grid_sample_xy[mask, :]
#             batch_index = (
#                 torch.arange(max_batch)
#                 .unsqueeze(1)
#                 .repeat(1, grid_sample_xy.shape[0])
#                 .view(-1, 1)
#             )
#             self.grid_sample_xy = nn.Parameter(
#                 torch.cat(
#                     [
#                         batch_index,
#                         grid_sample_xy.unsqueeze(0).repeat(max_batch, 1, 1).view(-1, 2),
#                     ],
#                     dim=1,
#                 ).long(),
#                 requires_grad=False,
#             )
#             self.length_per = grid_sample_xy.shape[0]

#     def forward(self, polar_feat, ref_feat):
#         grid_sample_index = self.grid_sample_index.to(ref_feat.device)
#         grid_sample_xy = self.grid_sample_xy.to(ref_feat.device)

#         grid_feat = ref_feat.clone()
#         length = self.length_per * ref_feat.shape[0]
#         grid_feat[
#             grid_sample_xy[:length, 0],
#             :,
#             grid_sample_xy[:length, 1],
#             grid_sample_xy[:length, 2],
#         ] = (
#             F.grid_sample(
#                 polar_feat,
#                 grid_sample_index[: ref_feat.shape[0], ...],
#                 mode="bilinear",
#                 padding_mode="zeros",
#                 align_corners=True,
#             )
#             .permute(0, 2, 1, 3)
#             .contiguous()
#             .reshape(-1, ref_feat.shape[1])
#         )
#         return grid_feat


# class Cart2Polar(nn.Module):
#     def __init__(
#         self, polar_size, cart_size, center_drop_grid_size=3.0, max_batch=8, mode="in"
#     ):
#         super(Cart2Polar, self).__init__()
#         if mode == "in":
#             yy, xx = torch.meshgrid(
#                 torch.arange(polar_size[0]), torch.arange(polar_size[1])
#             )
#             theta = np.pi - xx / polar_size[1] * np.pi * 2.0

#             index_x = (polar_size[0] - 0.5 - yy + center_drop_grid_size) * torch.cos(
#                 theta
#             ) / (polar_size[0] + center_drop_grid_size) * cart_size[
#                 0
#             ] / 2.0 + cart_size[
#                 0
#             ] / 2.0
#             index_y = (polar_size[0] - 0.5 - yy + center_drop_grid_size) * torch.sin(
#                 theta
#             ) / (polar_size[0] + center_drop_grid_size) * cart_size[
#                 0
#             ] / 2.0 + cart_size[
#                 0
#             ] / 2.0

#             grid_sample_index = (
#                 torch.stack((index_x, index_y), axis=-1)
#                 .view(-1, 2)
#                 .unsqueeze(0)
#                 .unsqueeze(2)
#                 / cart_size[0]
#                 * 2.0
#                 - 1.0
#             )
#             self.grid_sample_index = nn.Parameter(
#                 grid_sample_index.repeat(max_batch, 1, 1, 1), requires_grad=False
#             )

#             grid_sample_xy = torch.stack((yy, xx), axis=-1).view(-1, 2)
#             batch_index = (
#                 torch.arange(max_batch)
#                 .unsqueeze(1)
#                 .repeat(1, grid_sample_xy.shape[0])
#                 .view(-1, 1)
#             )
#             self.grid_sample_xy = nn.Parameter(
#                 torch.cat(
#                     [
#                         batch_index,
#                         grid_sample_xy.unsqueeze(0).repeat(max_batch, 1, 1).view(-1, 2),
#                     ],
#                     dim=1,
#                 ).long(),
#                 requires_grad=False,
#             )
#             self.length_per = grid_sample_xy.shape[0]

#     def forward(self, grid_feat, ref_feat):
#         grid_sample_index = self.grid_sample_index.to(ref_feat.device)
#         grid_sample_xy = self.grid_sample_xy.to(ref_feat.device)

#         polar_feat = ref_feat.clone()
#         length = self.length_per * ref_feat.shape[0]

#         polar_feat[
#             grid_sample_xy[:length, 0],
#             :,
#             grid_sample_xy[:length, 1],
#             grid_sample_xy[:length, 2],
#         ] = (
#             F.grid_sample(
#                 grid_feat,
#                 grid_sample_index[: ref_feat.shape[0], ...],
#                 mode="bilinear",
#                 padding_mode="zeros",
#                 align_corners=True,
#             )
#             .permute(0, 2, 1, 3)
#             .contiguous()
#             .reshape(-1, ref_feat.shape[1])
#         )
#         return polar_feat

# utils/polar_cartesian_final.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rep(t: torch.Tensor, B: int):
    """(N,2) → (B,N,1,2) 로 batch 차원 복제"""
    return t.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)


# ─────────────────── Polar ➜ Cartesian ────────────────────
class Polar2Cart(nn.Module):
    """
    polar_feat : (B,C,Hp,Wp)   (row=θ, col=φ)  φ=0 ⇒ +x  (오른쪽)
    ref_cart   : (B,C,Hc,Wc)
    """

    def __init__(self, polar_size, cart_size, max_batch: int = 8):
        super().__init__()
        Hp, Wp = polar_size
        Hc, Wc = cart_size
        r_max = Hc / 2 - 0.5  # 공통 반경 기준

        # Cartesian grid 좌표
        yy, xx = torch.meshgrid(torch.arange(Hc), torch.arange(Wc))
        y = yy - r_max
        x = xx - r_max

        r = torch.sqrt(x**2 + y**2)  # 0 … r_max√2
        phi = (torch.atan2(y, x) + 2 * np.pi) % (2 * np.pi)  # 0 … 2π

        idx_row = torch.clamp(r / r_max, 0, 1) * (Hp - 1)  # θ
        # φ 축: col=0이 **왼쪽**, col=Wp-1이 **오른쪽(+x)**
        idx_col = (Wp - 1) - phi / (2 * np.pi) * (Wp - 1)

        mask = r <= r_max
        grid = torch.stack((idx_col, idx_row), -1)[mask]  # (N,2)
        grid[..., 0] = grid[..., 0] / Wp * 2 - 1  # x 정규화
        grid[..., 1] = grid[..., 1] / Hp * 2 - 1  # y 정규화
        self.register_buffer("grid", _rep(grid, max_batch), False)

        yy_mask, xx_mask = torch.where(mask)
        dst_xy = torch.stack((yy_mask, xx_mask), -1)
        bid = torch.arange(max_batch).unsqueeze(1).repeat(1, dst_xy.size(0))
        self.register_buffer(
            "oidx",
            torch.cat((bid.unsqueeze(-1), dst_xy.repeat(max_batch, 1, 1)), -1).view(-1, 3).long(),
            False,
        )

    def forward(self, polar_feat: torch.Tensor, ref_cart: torch.Tensor):
        B, C, _, _ = ref_cart.shape
        grid = self.grid[:B].to(ref_cart.device)
        oidx = self.oidx[: B * grid.size(1)].to(ref_cart.device)

        cart = ref_cart.clone()
        sampled = F.grid_sample(polar_feat, grid, align_corners=True).permute(0, 2, 1, 3).reshape(-1, C)
        cart[oidx[:, 0], :, oidx[:, 1], oidx[:, 2]] = sampled
        return cart


# ─────────────────── Cartesian ➜ Polar ────────────────────
class Cart2Polar(nn.Module):
    """
    cart_feat : (B,C,Hc,Wc)
    ref_polar : (B,C,Hp,Wp)
    """

    def __init__(self, polar_size, cart_size, max_batch: int = 8):
        super().__init__()
        Hp, Wp = polar_size
        Hc, Wc = cart_size
        r_max = Hc / 2 - 0.5

        # Polar grid 좌표
        row, col = torch.meshgrid(torch.arange(Hp), torch.arange(Wp))
        # φ 방향을 Polar2Cart 와 동일하게: col=0(왼) → col=Wp-1(오른, +x)
        phi = (Wp - 1 - col) / (Wp - 1) * 2 * np.pi  # 0 … 2π
        r = row / (Hp - 1) * r_max

        idx_x = r * torch.cos(phi) + r_max
        idx_y = r * torch.sin(phi) + r_max

        grid = torch.stack((idx_x, idx_y), -1).view(-1, 2)
        grid[..., 0] = grid[..., 0] / Wc * 2 - 1
        grid[..., 1] = grid[..., 1] / Hc * 2 - 1
        self.register_buffer("grid", _rep(grid, max_batch), False)

        rc = torch.stack((row, col), -1).view(-1, 2)
        bid = torch.arange(max_batch).unsqueeze(1).repeat(1, rc.size(0))
        self.register_buffer(
            "oidx",
            torch.cat((bid.unsqueeze(-1), rc.repeat(max_batch, 1, 1)), -1).view(-1, 3).long(),
            False,
        )

    def forward(self, cart_feat: torch.Tensor, ref_polar: torch.Tensor):
        B, C, _, _ = ref_polar.shape
        grid = self.grid[:B].to(ref_polar.device)
        oidx = self.oidx[: B * grid.size(1)].to(ref_polar.device)

        polar = ref_polar.clone()
        sampled = F.grid_sample(cart_feat, grid, align_corners=True).permute(0, 2, 1, 3).reshape(-1, C)
        polar[oidx[:, 0], :, oidx[:, 1], oidx[:, 2]] = sampled
        return polar
