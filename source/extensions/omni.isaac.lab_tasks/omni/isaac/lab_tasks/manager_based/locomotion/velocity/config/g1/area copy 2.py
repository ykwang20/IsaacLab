import torch
import math
import time 

class CuboidCrossSection:
    """
    Quickly compute the (n,k) tensor of cross-sectional areas for k identical
    cuboids in n environments, fully on GPU.  All geometry params are set once
    at construction time.
    """
    def __init__(self,
                 Lx: float = 0.20311,
                 Ly: float = 0.06547,
                 Lz: float = 0.01851,
                 rO_local = (-0.03592, 0.0, 0.02517),
                 device   = "cuda",
                 dtype    = torch.float64):
        # ----- geometry -----
        self.hx = Lx / 2.0
        self.hy = Ly / 2.0
        self.hz = Lz / 2.0
        self.rO = torch.as_tensor(rO_local, device=device, dtype=dtype)

        # 8 local vertices  (1,8,3)  -> kept as buffer
        self.v_local = torch.tensor([[sx*self.hx, sy*self.hy, sz*self.hz]
                                     for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],
                                    device=device, dtype=dtype)          # (8,3)

        # 12 edge index pairs (1,12,2)
        self.edges   = torch.tensor([[0,1],[0,2],[0,4], [1,3],[1,5], [2,3],[2,6],
                                     [3,7], [4,5],[4,6],[5,7],[6,7]],
                                    device=device, dtype=torch.long)     # (12,2)

        self.device, self.dtype = device, dtype

    # ---------- static helpers ----------
    @staticmethod
    def _R_zyx_batch(euler: torch.Tensor) -> torch.Tensor:
        """
        euler (...,3)  ->  R (...,3,3)    (roll-pitch-yaw, z-y-x order)
        """
        roll, pitch, yaw = euler.unbind(-1)
        cr, sr = roll.cos(),  roll.sin()
        cp, sp = pitch.cos(), pitch.sin()
        cy, sy = yaw.cos(),   yaw.sin()

        R = torch.empty(euler.shape[:-1] + (3,3), dtype=euler.dtype, device=euler.device)
        R[...,0,0] = cy*cp
        R[...,0,1] = cy*sp*sr - sy*cr
        R[...,0,2] = cy*sp*cr + sy*sr
        R[...,1,0] = sy*cp
        R[...,1,1] = sy*sp*sr + cy*cr
        R[...,1,2] = sy*sp*cr - cy*sr
        R[...,2,0] = -sp
        R[...,2,1] = cp*sr
        R[...,2,2] = cp*cr
        return R

    @staticmethod
    def _R_from_quat_batch(q: torch.Tensor) -> torch.Tensor:
        """
        q : (...,4)  [w, x, y, z]，可非单位；返回 (...,3,3)
        公式按右手坐标系、列向量形式推导
        """
        w, x, y, z = q.unbind(-1)

        # 归一化（防止采样误差）
        norm = torch.rsqrt(w*w + x*x + y*y + z*z + 1e-12)
        w *= norm; x *= norm; y *= norm; z *= norm

        # 复用中间项，减少乘法
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = torch.empty(q.shape[:-1] + (3,3), dtype=q.dtype, device=q.device)
        R[...,0,0] = 1 - 2*(yy + zz)
        R[...,0,1] =     2*(xy - wz)
        R[...,0,2] =     2*(xz + wy)

        R[...,1,0] =     2*(xy + wz)
        R[...,1,1] = 1 - 2*(xx + zz)
        R[...,1,2] =     2*(yz - wx)

        R[...,2,0] =     2*(xz - wy)
        R[...,2,1] =     2*(yz + wx)
        R[...,2,2] = 1 - 2*(xx + yy)
        return R


    # ---------- public API ----------
    def cross_area_batch(self,
                         O_world: torch.Tensor,        # (n,k,3)
                         quat:   torch.Tensor,        # (n,k,3)
                         h_plane: float | torch.Tensor # scalar, (n,) or (n,k)
                         ):
        """
        Returns (areas, elapsed_ms)
        areas : (n,k) tensor on same device / dtype as O_world
        """
        n, k, _ = O_world.shape
        B       = n * k
        dev     = self.device
        dt      = self.dtype

        # flatten
        Of = O_world.reshape(B,3)
        Quat =quat.reshape(B,4)

        # rotation & center
        R  = self._R_from_quat_batch(Quat)
        Cf = Of - (R @ self.rO)                          # (B,3)

        # world vertices   (B,8,3)
        verts_w = torch.einsum('bij,vj->bvi', R, self.v_local) + Cf[:,None,:]

        # edges
        verts_pair = verts_w[:, self.edges]              # (B,12,2,3)
        z_pair     = verts_pair[...,2]                   # (B,12,2)

        # broadcast h_plane to (B,)
        h_plane = torch.as_tensor(h_plane, device=dev, dtype=dt)
        if h_plane.ndim == 0:
            hb = h_plane.expand(B)
        else:
            hb = h_plane.reshape(-1) if h_plane.numel()==B else h_plane.repeat_interleave(k)
            hb = hb.to(device=dev, dtype=dt)

        side     = z_pair - hb[:,None,None]              # (B,12,2)
        straddle = (side[...,0]*side[...,1]) < 0         # (B,12)

        # t & intersection points
        t        = (hb[:,None] - z_pair[...,0]) / (z_pair[...,1]-z_pair[...,0] + 1e-12)
        t        = t.clamp(0,1)
        inter_xy = verts_pair[...,0,:2] + t.unsqueeze(-1) * (verts_pair[...,1,:2]-verts_pair[...,0,:2])
        inter_xy[~straddle] = float('nan')

        # --- GPU vectorised centroid, polar sort, shoelace ---

        mask      = torch.isfinite(inter_xy[...,0])           # (B,12)
        valid_cnt = mask.sum(dim=1)                           # (B,)

        # 质心
        centroid  = (torch.where(mask.unsqueeze(-1), inter_xy, 0.)
                     .sum(dim=1) / valid_cnt.clamp(min=1).unsqueeze(-1))   # (B,2)

        dx = inter_xy[...,0] - centroid[:,0,None]
        dy = inter_xy[...,1] - centroid[:,1,None]
        angles = torch.atan2(dy, dx)
        angles[~mask] = float('inf')                          # 无效点排到末尾

        idx         = torch.argsort(angles, dim=1)            # (B,12)
        pts_sorted  = torch.gather(inter_xy, 1, idx.unsqueeze(-1).expand(-1,-1,2))
        mask_sorted = torch.gather(mask,      1, idx)         # 有效 True 全在前面

        # 真实每批顶点数 m 以及全局最大 m_max (≤6)
        m      = mask_sorted.sum(dim=1)                       # (B,)
        m_max  = int(m.max().item())
        if m_max == 0:
            areas = torch.zeros(B, device=inter_xy.device, dtype=inter_xy.dtype)
            return areas.view(n,k)

        # 取前 m_max 个位置的坐标 (不足 m_max 的行后面是无效点，但我们会屏蔽)
        pts_m  = pts_sorted[:, :m_max, :]                     # (B, m_max, 2)

        x = pts_m[...,0]                                      # (B, m_max)
        y = pts_m[...,1]

        # 构造 next 索引：对每一行 i, next_j = (j+1) % m_i
        arange = torch.arange(m_max, device=x.device).unsqueeze(0)          # (1, m_max)
        m_b    = m.unsqueeze(1).clamp(min=1)                                # (B,1)
        next_idx = (arange + 1) % m_b                                       # (B, m_max)

        # 按行 gather 下一个顶点
        x_next = torch.gather(x, 1, next_idx)
        y_next = torch.gather(y, 1, next_idx)

        # 只保留 j < m_i 的位置
        valid_pos = arange < m_b                                            # (B, m_max)

        cross = (x * y_next - y * x_next)
        cross = torch.where(valid_pos, cross, torch.zeros_like(cross))

        areas = 0.5 * cross.sum(dim=1).abs()
        areas[m < 3] = 0.   # 少于3点无面积

        return areas.view(n, k)
    
def cpu_reference_areas(O_world, quat, h_plane,
                        Lx=0.20311, Ly=0.06547, Lz=0.01851,
                        rO_local=(-0.03592,0.0,0.02517),
                        dtype=torch.float64):
    """
    O_world: (B,3)  ; quat: (B,4) ; h_plane: 标量或 (B,)
    返回: (B,) CPU 双精度基准
    """
    device = 'cpu'
    B = O_world.shape[0]
    O = O_world.to(device, dtype)
    q = quat.to(device, dtype)

    # 归一化
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w,x,y,z = q.unbind(-1)
    # 旋转矩阵
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x),   1-2*(x*x+y*y)], -1),
    ], -2)  # (B,3,3)

    hx, hy, hz = Lx/2, Ly/2, Lz/2
    v_local = torch.tensor([[sx*hx, sy*hy, sz*hz]
                            for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],
                            dtype=dtype, device=device)   # (8,3)
    rO = torch.tensor(rO_local, dtype=dtype, device=device)

    # 中心
    C = O - (R @ rO)   # (B,3)

    verts = (R @ v_local.T).transpose(1,2) + C[:,None,:]  # (B,8,3)

    edges = torch.tensor([
        [0,1],[0,2],[0,4], [1,3],[1,5], [2,3],[2,6],
        [3,7], [4,5],[4,6],[5,7],[6,7]
    ], device=device)

    if not torch.is_tensor(h_plane):
        h_arr = torch.full((B,), float(h_plane), dtype=dtype, device=device)
    else:
        h_arr = h_plane.to(device, dtype).flatten()
        if h_arr.numel()==1:
            h_arr = h_arr.expand(B)

    areas = torch.zeros(B, dtype=dtype)
    for b in range(B):
        pts = []
        for e0,e1 in edges:
            a = verts[b,e0]; c = verts[b,e1]
            z0, z1 = a[2].item(), c[2].item()
            h = h_arr[b].item()
            prod = (z0-h)*(z1-h)
            if prod < 0:  # straddle
                t = (h - z0) / (z1 - z0)
                p = a + t*(c-a)
                pts.append(p[:2])
        if len(pts) >= 3:
            pts = torch.stack(pts)  # (m,2)
            center = pts.mean(0)
            ang = torch.atan2(pts[:,1]-center[1], pts[:,0]-center[0])
            order = torch.argsort(ang)
            pts = pts[order]
            x = pts[:,0]; y = pts[:,1]
            area = 0.5*abs((x*torch.roll(y,-1) - y*torch.roll(x,-1)).sum()).item()
            areas[b] = area
    return areas


cs = CuboidCrossSection(device="cuda", dtype=torch.float64)

n, k = 2, 2
# O  = torch.rand(n, k, 3, device="cuda", dtype=torch.float64)*2
# E  = torch.randn(n, k, 4, device="cuda", dtype=torch.float64)*0.4
# O = torch.tensor([[[1.,1.,-0.01], [3.,2.,0.14]],[[3.,1., 0.2],[3.,3.,0.04]]], device="cuda",dtype=float)
# E   = torch.tensor([[[1.,0,0,0], [1.,0,0,0]],[[1.,0,0,0],[1.,0,0,0]]], device="cuda",dtype=float)    # 小角度随机
# h  = 0.012
# start=time.time()
# areas = cs.cross_area_batch(O, E, h)
# end=time.time()
# print("areas=", areas)        # torch.Size([4096, 2])
# print(f"GPU time: {(end-start)*24:.3f} s")

cs = CuboidCrossSection(device="cuda", dtype=torch.float64)

O = torch.tensor([
    [1.0, 1.0, -0.01],   # 全部在截面上方 -> 0
    [3.0, 2.0,  0.14],   # 截面在其下 -> 0
    [2.5, 1.2,  0.04],   # 截面穿过 -> 满矩形
    [0.2, 0.3,  0.022]   # 截面穿过 -> 满矩形
], device="cuda",dtype=cs.dtype)

Q = torch.tensor([[1,0,0,0]]*4, device="cuda",dtype=cs.dtype)   # (4,4)
h = 0.012

areas_gpu = cs.cross_area_batch(O.view(2,2,3), Q.view(2,2,4), h).view(-1)
areas_ref = cpu_reference_areas(O, Q, h)
print("Test1 GPU:", areas_gpu)
print("Test1 REF:", areas_ref)
# 期望： [0, 0, 0.0132976, 0.0132976]


yaw = math.radians(45)
# 四元数 (w, x, y, z) for yaw about z
cy2 = math.cos(yaw/2); sy2 = math.sin(yaw/2)
Q_yaw = torch.tensor([[cy2, 0.0, 0.0, sy2]]*2, device="cuda",dtype=cs.dtype)
O2 = torch.tensor([
    [0.0, 0.0, 0.04],   # inside
    [0.0, 0.0, 0.20],   # above slab -> 0
], device="cuda",dtype=cs.dtype)
areas_gpu2 = cs.cross_area_batch(O2.view(1,2,3), Q_yaw.view(1,2,4), h).view(-1)
areas_ref2 = cpu_reference_areas(O2, Q_yaw, h)
print("Test2 GPU:", areas_gpu2)
print("Test2 REF:", areas_ref2)
# 第一个 = Lx*Ly, 第二个 = 0


# 选一个中心 Cz，使 bottom = h  或 top = h
Lz = 0.01851; hz = Lz/2
# 我们知道 Cz - hz = h  -> Cz = h + hz
Cz_touch_bottom = h + hz
# O_z = Cz + 0.02517  (因为 Cz = O_z - 0.02517)
O_bottom_tangent_z = Cz_touch_bottom + 0.02517

Cz_touch_top = h - hz
O_top_tangent_z = Cz_touch_top + 0.02517

O3 = torch.tensor([
    [0,0,O_bottom_tangent_z],  # 平面 = bottom
    [0,0,O_top_tangent_z]      # 平面 = top
], device="cuda",dtype=cs.dtype)
Q_id = torch.tensor([[1,0,0,0],[1,0,0,0]],device="cuda", dtype=cs.dtype)

areas_gpu3 = cs.cross_area_batch(O3.view(1,2,3), Q_id.view(1,2,4), h).view(-1)
print("Test3 boundary areas:", areas_gpu3)  # 期望 tensor([0., 0.])


def quat_from_axis_angle(axis, angle):
    axis = torch.tensor(axis, dtype=cs.dtype)
    axis = axis / axis.norm()
    s = math.sin(angle/2)
    return torch.tensor([math.cos(angle/2),
                         axis[0]*s, axis[1]*s, axis[2]*s], device="cuda",dtype=cs.dtype)

pitch = math.radians(30)   # 绕 x
Q_pitch = quat_from_axis_angle([1,0,0], pitch).unsqueeze(0)  # (1,4)

# 3 个不同 O_z
O4 = torch.tensor([
    [0,0, 0.02],   # plane 大概在中下部
    [0,0, 0.04],   # plane 处于中间
    [0,0, 0.06],   # plane 靠上
], device="cuda",dtype=cs.dtype)

areas_gpu4 = cs.cross_area_batch(O4.view(1,3,3), Q_pitch.view(1,1,4).expand(1,3,4), h).view(-1)
areas_ref4 = cpu_reference_areas(O4, Q_pitch.expand(3,4), h)
print("Test4 GPU:", areas_gpu4)
print("Test4 REF:", areas_ref4)
# 这里三个值都 <= 满矩形面积，且中间那个通常最大；数值需与 ref 近似一致


torch.manual_seed(0)
B_rand = 1000
n, k = 100, 10
# 随机 O (控制 z 让一部分有交集)
O_rand = torch.zeros(n,k,3, dtype=cs.dtype, device="cuda")
O_rand[...,0] = torch.rand(n,k, device="cuda") * 0.5
O_rand[...,1] = torch.rand(n,k, device="cuda") * 0.5
O_rand[...,2] = torch.rand(n,k, device="cuda") * 0.25   # 高度 0~0.25

# 随机四元数
q_raw = torch.randn(n,k,4, device="cuda", dtype=cs.dtype)
q_rand = q_raw / q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-12)

h_rand = 0.012
areas_gpu5 = cs.cross_area_batch(O_rand, q_rand, h_rand)   # (n,k)
# 取部分样本送入 CPU 参考
sel = torch.randperm(n*k)[:64]
areas_ref5 = cpu_reference_areas(O_rand.view(-1,3)[sel].cpu(),
                                 q_rand.view(-1,4)[sel].cpu(),
                                 h_rand)
diff = (areas_gpu5.view(-1)[sel].cpu() - areas_ref5).abs()
print("Test5 max diff:", diff.max())
print("Any NaN GPU?:", torch.isnan(areas_gpu5).any().item())
# 期望 max diff ~ 1e-9 (float64)；无 NaN

