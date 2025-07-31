import cv2
import numpy as np
import matplotlib.pyplot as plt
# mm pour chaque pixel -> issu du .cfg des cameras
MM_PER_PIXEL = 0.0287

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

def magnitude_map(flow, mm_per_pixel=MM_PER_PIXEL):
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude * mm_per_pixel


if __name__ == "__main__":
    img1 = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./data/left/img1.tiff", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Images non trouvées.")

    flow = motion_images(img1, img2)
    magnitude = magnitude_map(flow)

    # matplotlib pour afficher
    plt.figure(figsize=(10, 8))
    height, width = magnitude.shape
    extent = [0, width * MM_PER_PIXEL, height * MM_PER_PIXEL, 0]
    im = plt.imshow(magnitude, cmap="viridis", origin="upper", extent=extent)
    plt.title("Norme du champ de déplacement (Farneback)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude du déplacement (mm)")
    plt.tight_layout()
    plt.show()

    