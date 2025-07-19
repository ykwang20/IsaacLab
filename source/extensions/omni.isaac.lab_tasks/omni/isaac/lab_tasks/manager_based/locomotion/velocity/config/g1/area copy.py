# import torch
# import math
# import time
# import numpy as np

# class AreaCompute:

# # ---------- 固定几何 ----------
# hx, hy, hz = 0.101555, 0.032735, 0.009255              # half-lengths
# rO_local = torch.tensor([-0.03592, 0.0, 0.02517])       # 固连点 O 相对中心

# # 8×3 顶点表（局部坐标）
# v_local = torch.tensor([[sx*hx, sy*hy, sz*hz]
#                         for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])  # (8,3)
# torch.set_printoptions(precision=7, sci_mode=False)
# # 12×2 棱的端点下标
# edges = torch.tensor([
#     [0,1],[0,2],[0,4], [1,3],[1,5], [2,3],[2,6],
#     [3,7], [4,5],[4,6],[5,7],[6,7]
# ])                                                     # (12,2)

# def R_zyx_batch(euler: torch.Tensor) -> torch.Tensor:
#     """
#     euler  : (B,3)  roll(x), pitch(y), yaw(z)
#     return : (B,3,3)
#     """
#     roll, pitch, yaw = euler.unbind(-1)
#     cr, sr = roll.cos(),  roll.sin()
#     cp, sp = pitch.cos(), pitch.sin()
#     cy, sy = yaw.cos(),   yaw.sin()

#     Rz = torch.stack([torch.stack([ cy, -sy, torch.zeros_like(cy)], -1),
#                       torch.stack([ sy,  cy, torch.zeros_like(cy)], -1),
#                       torch.tensor([0.,0.,1.], device=euler.device).repeat(euler.size(0),1)], -2)

#     Ry = torch.stack([torch.stack([ cp, torch.zeros_like(cp),  sp], -1),
#                       torch.stack([ torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp) ], -1),
#                       torch.stack([-sp, torch.zeros_like(cp),  cp], -1)], -2)

#     Rx = torch.stack([torch.stack([ torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr) ], -1),
#                       torch.stack([ torch.zeros_like(cr),  cr, -sr], -1),
#                       torch.stack([ torch.zeros_like(cr),  sr,  cr], -1)], -2)

#     return Rz @ Ry @ Rx           # (B,3,3)

# def cross_area_batch(O_world, euler, h_plane):
#     """
#     O_world : (n,k,3)  — O 点世界坐标
#     euler   : (n,k,3)  — roll, pitch, yaw  (弧度, z-y-x 顺序)
#     h_plane : (n,) 或标量  — 各环境的截面高度 (可一律相同)

#     return  : (n,k) 截面积
#     """
#     n, k, _ = O_world.shape
#     device  = O_world.device
#     B       = n * k                                   # 展平后的总 batch

#     # 展平 batch
#     O_flat   = O_world.reshape(B, 3)
#     e_flat   = euler.reshape(B, 3)

#     # (B,3,3) 旋转矩阵
#     R        = R_zyx_batch(e_flat)

#     # 中心坐标  C = O - R @ rO_local
#     C_flat   = O_flat - (R @ rO_local.to(device))     # (B,3)

#     # 世界坐标 8 个顶点：einsum 做批量矩阵乘
#     verts_w  = torch.einsum('bij,vj->bvi', R, v_local.to(device)) + C_flat[:,None,:]   # (B,8,3)

#     # 取两个端点组成棱  → (B,12,2,3)
#     verts_pair = verts_w[:, edges]                                   # (B,12,2,3)
#     z_pair     = verts_pair[..., 2]                                  # (B,12,2)

#     # h_plane 展平后与 batch 对齐
#     if torch.is_tensor(h_plane) and h_plane.ndim==1:
#         h_b = h_plane.repeat_interleave(k).to(device)                # (B,)
#     else:
#         h_b = torch.full((B,), float(h_plane), device=device)

#     # 判断跨越平面 (z-h)*(z'-h)<0
#     side      = z_pair - h_b[:,None,None]                            # (B,12,2)
#     straddle  = (side[...,0] * side[...,1]) < 0                      # (B,12)

#     # 计算插值系数 t=(h-za)/(zb-za)
#     t         = (h_b[:,None] - z_pair[...,0]) / (z_pair[...,1]-z_pair[...,0] + 1e-12)
#     t         = t.clamp(0,1)                                         # (B,12)

#     # 得到交点，先全部算，再用 mask 无效化
#     inter_pts = verts_pair[...,0,:] + t.unsqueeze(-1) * (verts_pair[...,1,:]-verts_pair[...,0,:])  # (B,12,3)
#     inter_pts_xy = inter_pts[...,:2]
#     inter_pts_xy[~straddle] = float('nan')

#     # ---------- GPU 向量化极角 + 鞋带公式 (无 for‑loop) ----------
#     t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
#     t0.record()

#     mask          = torch.isfinite(inter_pts_xy[..., 0])          # (B,12)
#     valid_cnt     = mask.sum(dim=1).clamp(min=1)                  # (B,)

#     # 质心
#     sum_xy        = torch.where(mask.unsqueeze(-1), inter_pts_xy, 0.).sum(dim=1)  # (B,2)
#     centroid      = sum_xy / valid_cnt.unsqueeze(-1)                               # (B,2)

#     # 极角；无效点设 +inf 让它排最后
#     dx = inter_pts_xy[..., 0] - centroid[:, 0].unsqueeze(1)
#     dy = inter_pts_xy[..., 1] - centroid[:, 1].unsqueeze(1)
#     angles        = torch.atan2(dy, dx)
#     angles[~mask] = float('inf')

#     # 排序并收集
#     idx           = torch.argsort(angles, dim=1)                                    # (B,12)
#     pts_sorted    = torch.gather(inter_pts_xy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
#     mask_sorted   = torch.gather(mask,           1, idx)

#     # 鞋带公式（只累加有效边）
#     x,  y         = pts_sorted[..., 0], pts_sorted[..., 1]
#     x_n, y_n      = torch.roll(x, -1, dims=1), torch.roll(y, -1, dims=1)
#     edge_mask     = mask_sorted & torch.roll(mask_sorted, -1, dims=1)

#     cross         = (x * y_n - y * x_n) * edge_mask
#     areas         = 0.5 * cross.sum(dim=1).abs()                                     # (B,)
#     areas[valid_cnt < 3] = 0.                                                        # m<3 ⇒ 0

#     t1.record(); torch.cuda.synchronize()
#     duration_ms   = t0.elapsed_time(t1)

#     return areas.view(n, k), duration_ms

# # 假设 n=32 环境, k=4 个长方体
# n, k = 4096, 2
# device = 'cuda:0' 

# # O_world = torch.tensor([[[1,1,-0.01], [3,2,0.14]],[[3,1, 0.2],[3,3,0.04]]], device=device)
# # euler   = torch.tensor([[[torch.pi,0,0], [3,2,0.14]],[[3,1, 0.2],[3,3,0.04]]], device=device)    # 小角度随机
# O_world = torch.rand(n, k, 3, device=device) * 2.0      # [0,2) 任选
# euler   = (torch.randn(n, k, 3, device=device) * 0.3)    # 小角度随机
# h_plane = 0.01

# start_time=time.time()

# areas, duration_ms = cross_area_batch(O_world, euler, h_plane)        # (n,k)
# end_time=time.time()
# print(areas)  # torch.Size([32, 4])
# print(f"The loop took {24*duration_ms:.4f} milliseconds for iterations.")
# print(24*(end_time-start_time))
