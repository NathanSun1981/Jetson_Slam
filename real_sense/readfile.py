import open3d as o3d

pc = o3d.io.read_point_cloud('/mnt/Git/Jetson_Slam/real_sense/temp/output.ply')
o3d.visualization.draw_geometries([pc])
