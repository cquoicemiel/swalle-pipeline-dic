import open3d as o3d

# Modifie ici le nom du fichier à afficher :
ply_path = ".\sparse_pointcloud_img0.ply"

pcd = o3d.io.read_point_cloud(ply_path)
print(pcd)
o3d.visualization.draw_geometries([pcd], window_name="Nuage de points 3D")