import open3d as o3d
import time
import os

# 📁 Dossier contenant les 31 fichiers .ply
ply_dir = "./output_sparse"
ply_files = sorted([
    os.path.join(ply_dir, f) for f in os.listdir(ply_dir)
    if f.endswith(".ply")
])

# Vérification
if not ply_files:
    print("❌ Aucun fichier .ply trouvé.")
    exit()

# 🖼️ Fenêtre de visualisation
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Animation nuage 3D Swall-E", width=1280, height=720)
geom_added = False

# 🔁 Animation
for path in ply_files:
    pcd = o3d.io.read_point_cloud(path)

    if not geom_added:
        vis.add_geometry(pcd)
        geom_added = True
    else:
        vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    print(f"🌀 Frame: {os.path.basename(path)}")
    time.sleep(0.2)  # Vitesse de l’animation (0.2s entre les frames)

# 🔚 Fin
vis.destroy_window()
