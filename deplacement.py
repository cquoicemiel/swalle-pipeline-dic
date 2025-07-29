import cv2
import numpy as np

def motion_images(img1, img2):

    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,           # plus petit => plus réactif aux petits détails
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0  # pas OPTFLOW_GAUSSIAN
    )

    return flow


def motion_to_hsv(flow, img): # mm echelle pour toutes les frames
    
    # conversion du champ de déplacement en image couleur (hsv)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2     # Hue (direction)
    hsv[..., 1] = 255                        # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (intensité)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_rgb

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


if __name__ == "__main__":
    
    
    img1 = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./data/right/img1.tiff", cv2.IMREAD_GRAYSCALE)
    
    flow_with_wheel = draw_direction_wheel(motion_to_hsv(motion_images(img1, img2), img1), radius=80, position=(100, 100))
    cv2.imshow("Déplacement", flow_with_wheel) # affiche le déplament avec le roue hsv en overlay
    cv2.waitKey(0)
    cv2.destroyAllWindows()
