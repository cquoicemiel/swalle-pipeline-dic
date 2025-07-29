import cv2
import numpy as np
import os
import open3d as o3d



# CONFIG
# Caméra principale (gauche)
K_left = np.array([[10731.6, 0, 1257.89],
                   [0, 10731.8, 1010.97],
                   [0, 0, 1]])
D_left = np.array([-0.0449469, 1.2968, 0.000686857, 0.000534226, 0])

# Caméra secondaire (droite)
K_right = np.array([[10776.4, 0, 1223.65],
                    [0, 10781, 997.056],
                    [0, 0, 1]])
D_right = np.array([-0.0388554, 0.702762, 0, 0, 0])

# Rotation et translation
R = np.array([[0.863015, -0.00155128, -0.505176],
              [0.000800376, 0.999998, -0.00170345],
              [0.505178, 0.00106577, 0.863014]])
T = np.array([307.603, 0.932237, 77.9891]) / 1000.0  # converesion en mètre

# TRAITEMENT
def process_pair(left_path, right_path, output_path):
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    h, w = imgL.shape

    # rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right,
                                                (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, (w, h), cv2.CV_32FC1)

    rect_left = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # disparité (bloquage ici)
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=64,
                                   blockSize=5,
                                   P1=8*3*5**2,
                                   P2=32*3*5**2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1

    # reconstruction en 3d
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = (disparity > 0.1) & np.isfinite(disparity)

    img_color = cv2.imread(left_path)
    points = points_3D[mask]
    colors = img_color[mask]

    if len(points) < 100:
        print(f"❌ Pas assez de points pour {left_path}")
        return

    # Exportation du "nuage" de points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✅ Exporté : {output_path}")

# BOUCLE

input_dir = "./data/" 
output_dir = "./output_ply"

for i in range(100):  # adapte à ton nombre de paires
    lpath = os.path.join(input_dir, f"left/img{i}.jpg")
    rpath = os.path.join(input_dir, f"right/img{i}.jpg")
    opath = os.path.join(output_dir, f"cloud_img{i}.ply")

    if os.path.exists(lpath) and os.path.exists(rpath):
        process_pair(lpath, rpath, opath)
