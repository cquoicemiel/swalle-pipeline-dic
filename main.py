import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

image_folder = "./droite"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tiff")))

images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

print(f"{len(images)} images chargées.")

img1 = images[1]
img2 = images[2]

flow = cv2.calcOpticalFlowFarneback(
    img1, img2,
    None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# 🎨 Convertir en magnitude et angle
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

hsv = np.zeros_like(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR))
hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (direction)
hsv[..., 1] = 255                     # Saturation
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (vitesse)
bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Image de départ")
plt.imshow(img1, cmap="gray")

plt.subplot(1,2,2)
plt.title("Champ de déplacement (Farneback)")
plt.imshow(bgr_flow)
plt.show()
