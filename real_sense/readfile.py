import open3d as o3d
import numpy as np

def is_black_point(x):
    return sum(x[3:6]) > 0

pc = o3d.io.read_point_cloud('/mnt/Git/Jetson_Slam/real_sense/output.pcd')
xyz = np.array(pc.points)
rgb = np.array(pc.colors)
rgb_index = np.array(list(map(sum,rgb)))

sub_pc = np.concatenate([xyz, rgb], axis=-1)

mask = (rgb_index != 0)
masked_pc = sub_pc[mask]

pc_new = o3d.geometry.PointCloud()
pc_new.points = o3d.utility.Vector3dVector(masked_pc[:, 0:3])
pc_new.colors = o3d.utility.Vector3dVector(masked_pc[:, 3:6])

o3d.visualization.draw_geometries([pc_new])
