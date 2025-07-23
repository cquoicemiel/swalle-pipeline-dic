import cv2
import numpy as np
from pretraitement import denoise, resize_to_fit

# chargement des images
img1 = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./data/right/img1.tiff", cv2.IMREAD_GRAYSCALE)

# on applique le pretraitement
img1_denoised = denoise(img1)
img2_denoised = denoise(img2)

# calcul du champ de déplacement (Optical Flow Farneback)
flow = cv2.calcOpticalFlowFarneback(
    img1_denoised, img2_denoised, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# conversion du champ de déplacement en image couleur (hsv)
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
hsv[..., 0] = angle * 180 / np.pi / 2     # Hue (direction)
hsv[..., 1] = 255                        # Saturation
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (intensité)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ajout de l'echelle au format roue de couleur rgb pour se référencer à ça
def draw_direction_wheel(img, radius=100, position=(120, 120)):
    wheel = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
    for y in range(2*radius):
        for x in range(2*radius):
            dx = x - radius
            dy = y - radius
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= radius:
                angle = np.arctan2(dy, dx)
                angle_deg = (angle * 180 / np.pi) / 2  # OpenCV hue range
                if angle_deg < 0:
                    angle_deg += 180
                wheel[y, x] = (angle_deg, 255, 255)
    wheel_bgr = cv2.cvtColor(wheel, cv2.COLOR_HSV2BGR)

    # Dessiner sur l’image principale
    x, y = position
    overlay = img.copy()
    overlay[y-radius:y+radius, x-radius:x+radius] = wheel_bgr
    return overlay

flow_with_wheel = draw_direction_wheel(flow_rgb, radius=80, position=(100, 100))





cv2.imshow("Déplacement (Optical Flow) + Échelle", resize_to_fit(flow_with_wheel))
cv2.waitKey(0)
cv2.destroyAllWindows()


if __name__ == "__main__":
    pass