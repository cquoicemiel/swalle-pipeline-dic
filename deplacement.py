import cv2
import numpy as np
import matplotlib.pyplot as plt

def motion_images(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def magnitude_map(flow):
    # norme quadra du vecteur (dx, dy)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    # flow est en 2d dont on fait 1ere dimension ^2 + 2eme dimensions ^2
    return magnitude

if __name__ == "__main__":
    img1 = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./data/right/img1.tiff", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Images non trouvées.")

    flow = motion_images(img1, img2)
    magnitude = magnitude_map(flow)

    # matplotlib pour afficher
    plt.figure(figsize=(10, 8))
    im = plt.imshow(magnitude, cmap="viridis", origin="upper")
    plt.title("Norme du champ de déplacement (Farneback)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude du déplacement (pixels)")
    plt.tight_layout()
    plt.show()

    