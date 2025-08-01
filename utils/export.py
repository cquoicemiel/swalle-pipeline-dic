import os
import cv2
import matplotlib.pyplot as plt
from pretraitement import denoise
from deplacement import motion_images, magnitude_map
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)

config = load_config()


def export_pngs(side, folder):
    mm_par_pixel =  float(config["calib_stereo"]["Ts1_12"])/float(config["cam_principale" if side == "left" else "cam_secondaire"]["fc1"])

    output_dir = f"{folder}/{side}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 30):
        a = cv2.imread(f"./data/{side}/img{i}.tiff", cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f"./data/{side}/img{i+1}.tiff", cv2.IMREAD_GRAYSCALE)

        if a is None or b is None:
            logger.warning(f"Image manquante à t={i}, sautée.")
            continue

        a_denoised = denoise(a)
        b_denoised = denoise(b)

        flow = motion_images(a_denoised, b_denoised)
        magnitude = magnitude_map(flow)  # mm

        height, width = magnitude.shape
        extent = [0, width * mm_par_pixel, height * mm_par_pixel, 0]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(magnitude, cmap="viridis", origin="upper", extent=extent)
        ax.set_title(f"Déplacement t{i} → t{i+1}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude du déplacement (mm)")

        output_path = f"{output_dir}/img{i}.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"{output_path} exporté.")

def export_animation(side, input_folder, output_folder):
    input_path = f"{input_folder}/{side}"
    output_path = f'{output_folder}/{side}_transitions.mp4'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
    if not images:
        logger.warning("aucune image trouvée dans", input_path)
        return

    logger.info(f"{len(images)} images trouvées.")

    first_frame = cv2.imread(os.path.join(input_path, images[0]))
    if first_frame is None:
        logger.error("erreur: premiere frame == None")
        return

    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            logger.error(f"image illisible : {img_name}")
            continue
        out.write(frame)

    out.release()
    logger.info("animation des frames exportée avec succès :", output_path)