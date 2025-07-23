import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img1 = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./data/left/img30.tiff", cv2.IMREAD_GRAYSCALE)

img1_denoised = cv2.bilateralFilter(img1, 9, 75, 75)
img2_denoised = cv2.bilateralFilter(img2, 9, 75, 75)

# Vérifier la taille
print("Image 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)

# # Afficher pour validation
# plt.subplot(1,2,1), plt.imshow(img1_denoised, cmap="gray"), plt.title("Angle 1")
# plt.subplot(1,2,2), plt.imshow(img2_denoised, cmap="gray"), plt.title("Angle 2")
# plt.show()

#////////////////////////////////////////////:

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img1_denoised, None)
kp2, des2 = orb.detectAndCompute(img2_denoised, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Afficher les meilleurs matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
plt.imshow(img_matches)
plt.show()