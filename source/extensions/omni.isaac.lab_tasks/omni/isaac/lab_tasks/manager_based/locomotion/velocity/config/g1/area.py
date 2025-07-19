import numpy as np

# ---------- 固定参数 ----------
hx, hy, hz = 0.101555, 0.032735, 0.009255            # half-lengths
rO_local = np.array([-0.03592, 0.0, 0.02517])        # O offset in local frame

# ---------- 帮助函数 ----------
def R_zyx(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx        # z-y-x order

def cross_area_O(O_world, euler, h_plane):
    roll, pitch, yaw = euler
    R = R_zyx(roll, pitch, yaw)

    # cuboid center in world
    C = O_world - R @ rO_local

    # 8 local vertices
    verts_local = np.array([[sx*hx, sy*hy, sz*hz]
                            for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    verts_world = (R @ verts_local.T).T + C

    # 12 edges by index
    edges = [(0,1),(0,2),(0,4), (1,3),(1,5), (2,3),(2,6),
             (3,7), (4,5),(4,6),(5,7),(6,7)]

    pts = []
    for i,j in edges:
        z1, z2 = verts_world[i,2], verts_world[j,2]
        if (z1 - h_plane)*(z2 - h_plane) < 0:          # crosses z=h
            t = (h_plane - z1) / (z2 - z1)
            p = verts_world[i] + t*(verts_world[j] - verts_world[i])
            pts.append(p[:2])

    if len(pts) < 3:
        return 0.0
    pts = np.array(pts)

    # polar sort
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-centroid[1], pts[:,0]-centroid[0])
    order = np.argsort(angles)
    pts = pts[order]

    # shoelace
    x, y = pts[:,0], pts[:,1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


O_world = np.array([1.5, 100, 0.04])   # 填入实际坐标
euler   = (0, 0, 0)          # 以弧度给出
h       = 0.012                        # 例：截面高度 5 cm
A = cross_area_O(O_world, euler, h)
print("截面面积 =", A, "m²")
