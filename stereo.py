import cv2
import numpy as np

fx1 = 10731.6
fy1 = 10731.8
cx1 = 1257.89
cy1 = 1010.97
ka1 = -0.0449469
ka2 = 1.2968
ka3 = 0.000686857

fx2 = 10776.4
fy2 = 10781
cx2 = 1223.65
cy2 = 997.056
kb1 = -0.0388554
kb2 = 0.702762
kb3 = 0

R = np.array([[0.863015, -0.00155128, -0.505176],
              [0.000800376, 0.999998, -0.00170345],
              [0.505178, 0.00106577, 0.863014]])

T = np.array([307.603, 0.932237, 77.9891]) / 1000.0  # mm → m

K_left = np.array([[fx1, 0, cx1],
                   [0, fy1, cy1],
                   [0, 0, 1]])
D_left = np.array([ka1, ka2, 0, 0, ka3])  # Pas de p1/p2

K_right = np.array([[fx2, 0, cx2],
                    [0, fy2, cy2],
                    [0, 0, 1]])
D_right = np.array([kb1, kb2, 0, 0, kb3])

img_left = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_UNCHANGED)
img_right = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_UNCHANGED)

# vérifier le chargement
if img_left is None or img_right is None:
    raise ValueError("❌ Impossible de charger les images. Vérifie les chemins et formats.")

#  Si TIFF 16 bits, normaliser en 8 bits
if img_left.dtype == np.uint16:
    img_left = cv2.convertScaleAbs(img_left, alpha=(255.0/65535.0))
if img_right.dtype == np.uint16:
    img_right = cv2.convertScaleAbs(img_right, alpha=(255.0/65535.0))

# 🖼️Taille réelle des images
image_size = (img_left.shape[1], img_left.shape[0])  # (largeur, hauteur)
print("✅ Taille image :", image_size)

# ♻️ Rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_left, D_left,
    K_right, D_right,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0  # Mettre 1 ou -1 si nécessaire
)

# Calcul des maps de correction
map1_left, map2_left = cv2.initUndistortRectifyMap(
    K_left, D_left, R1, P1, image_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    K_right, D_right, R2, P2, image_size, cv2.CV_16SC2
)

# 📸 Rectifier les images
rect_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
rect_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

# 🖋️ Tracer des lignes horizontales pour vérifier l’alignement
for y in range(0, image_size[1], 50):  # une ligne tous les 50 pixels
    cv2.line(rect_left, (0, y), (image_size[0], y), (255, 255, 255), 1)
    cv2.line(rect_right, (0, y), (image_size[0], y), (255, 255, 255), 1)

# 👁️ Afficher les images rectifiées
cv2.imshow("Rectified Left", rect_left)
cv2.imshow("Rectified Right", rect_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
