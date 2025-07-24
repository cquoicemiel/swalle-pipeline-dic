import cv2
import numpy as np
import open3d

# --- Calibration données ---
fx1, fy1, cx1, cy1 = 10731.6, 10731.8, 1257.89, 1010.97
fx2, fy2, cx2, cy2 = 10776.4, 10781, 1223.65, 997.056
ka1, ka2, ka3 = -0.0449469, 1.2968, 0.000686857
kb1, kb2, kb3 = -0.0388554, 0.702762, 0

R = np.array([[0.863015, -0.00155128, -0.505176],
              [0.000800376, 0.999998, -0.00170345],
              [0.505178, 0.00106577, 0.863014]])
T = np.array([307.603, 0.932237, 77.9891]) / 1000.0

K_left = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
D_left = np.array([ka1, ka2, 0, 0, ka3])
K_right = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
D_right = np.array([kb1, kb2, 0, 0, kb3])

# --- Lecture images ---
imgL = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_UNCHANGED)
imgR = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_UNCHANGED)

# --- Normalisation si 16 bits ---
if imgL.dtype == np.uint16:
    imgL = cv2.convertScaleAbs(imgL, alpha=255.0/65535.0)
if imgR.dtype == np.uint16:
    imgR = cv2.convertScaleAbs(imgR, alpha=255.0/65535.0)

# --- Taille image ---
image_size = (imgL.shape[1], imgL.shape[0])

# --- Rectification ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)
rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# --- Affichage lignes horizontales pour validation ---
rectL_color = cv2.cvtColor(rectifiedL, cv2.COLOR_GRAY2BGR)
rectR_color = cv2.cvtColor(rectifiedR, cv2.COLOR_GRAY2BGR)
for y in range(0, image_size[1], 50):
    cv2.line(rectL_color, (0, y), (image_size[0], y), (0, 255, 0), 1)
    cv2.line(rectR_color, (0, y), (image_size[0], y), (0, 255, 0), 1)
cv2.imshow("Rectified L", rectL_color)
cv2.imshow("Rectified R", rectR_color)
cv2.waitKey(0)

# --- Calcul disparité ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=9,
    P1=8*3*9**2,
    P2=32*3*9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

# --- Visualisation disparité ---
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
cv2.imshow("Disparity Map", disp_vis)
cv2.waitKey(0)

# --- Projection 3D ---
points_3D = cv2.reprojectImageTo3D(disparity, Q)
mask = disparity > disparity.min()
points = points_3D[mask]
colors = rectL_color[mask]

# --- Sauvegarde avec Open3D ---
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points)
pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
open3d.io.write_point_cloud("./output/pointcloud_t0.ply", pcd)

print("Disparity min/max:", np.min(disparity), np.max(disparity))
