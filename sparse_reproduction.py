import cv2
import numpy as np
import open3d as o3d
import os

# Calibration
K_left = np.array([[10731.6, 0, 1257.89],
                   [0, 10731.8, 1010.97],
                   [0, 0, 1]])
D_left = np.array([-0.0449469, 1.2968, 0, 0, 0.000686857])

K_right = np.array([[10776.4, 0, 1223.65],
                    [0, 10781, 997.056],
                    [0, 0, 1]])
D_right = np.array([-0.0388554, 0.702762, 0, 0, 0])

R = np.array([[0.863015, -0.00155128, -0.505176],
              [0.000800376, 0.999998, -0.00170345],
              [0.505178, 0.00106577, 0.863014]])
T = np.array([[307.603], [0.932237], [77.9891]]) / 1000.0

P1 = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K_right @ np.hstack((R, T))

os.makedirs("./output_sparse", exist_ok=True)

for t in range(31):
    imgL = cv2.imread(f"./data/left/img{t}.tiff", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(f"./data/right/img{t}.tiff", cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print(f"⚠️ Images t={t} manquantes, a été sautée.")
        continue

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    if des1 is None or des2 is None:
        print(f"⚠️ Pas de descripteurs à t={t}")
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    if len(matches) < 10:
        print(f"⚠️ Trop peu de matches à t={t}, sauté.")
        continue

    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    pts1_ud = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K_left, D_left)
    pts2_ud = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K_right, D_right)

    pts4D = cv2.triangulatePoints(P1, P2, pts1_ud, pts2_ud)
    pts3D = (pts4D[:3] / pts4D[3]).T

    # Couleurs depuis img gauche (cam principale)
    colors = np.array([imgL[int(y), int(x)] for x, y in pts1], dtype=np.uint8)
    colors_rgb = np.stack([colors] * 3, axis=1) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3D)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    out_path = f"./output_sparse/pointcloud_t{t:02d}.ply"
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"✅ t={t:02d} → {out_path} exporté.")
