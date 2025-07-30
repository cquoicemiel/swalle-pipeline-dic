import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pretraitement import denoise, resize_to_fit, link_matches, img_dim, diff
from deplacement import motion_images, magnitude_map
# from skimage.registration import optical_flow_tvl1

# img1 = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_GRAYSCALE)

# img1_denoised = denoise(img1)
# img2_denoised = denoise(img2)

# img1_denoised_color = cv2.cvtColor(img1_denoised, cv2.COLOR_GRAY2BGR)
# img2_denoised_color = cv2.cvtColor(img2_denoised, cv2.COLOR_GRAY2BGR)

# img3 = cv2.imread("./data/left/img2.tiff", cv2.IMREAD_GRAYSCALE)
# img4 = cv2.imread("./data/left/img3.tiff", cv2.IMREAD_GRAYSCALE)

# img3_denoised = denoise(img3)
# img4_denoised = denoise(img4)

# img3_denoised_color = cv2.cvtColor(img3_denoised, cv2.COLOR_GRAY2BGR)
# img4_denoised_color = cv2.cvtColor(img4_denoised, cv2.COLOR_GRAY2BGR)

# img5 = cv2.imread("./data/left/img4.tiff", cv2.IMREAD_GRAYSCALE)
# img6 = cv2.imread("./data/left/img5.tiff", cv2.IMREAD_GRAYSCALE)

# img5_denoised = denoise(img5)
# img6_denoised = denoise(img6)

# img5_denoised_color = cv2.cvtColor(img5_denoised, cv2.COLOR_GRAY2BGR)
# img6_denoised_color = cv2.cvtColor(img6_denoised, cv2.COLOR_GRAY2BGR)



# u, v = optical_flow_tvl1(img1_denoised, img2_denoised)

# plt.quiver(u[::10, ::10], -v[::10, ::10])
# plt.title("Champ de déplacement 2D (quiver)")
# plt.show()

# print(img_dim(img1))
# print(img_dim(img2))

# cv2.imshow("Images denoised", resize_to_fit(np.hstack((img1_denoised, img2_denoised))))

# cv2.imshow("Correspondances", resize_to_fit(link_matches(img1_denoised, img2_denoised)))

# cv2.imshow("Difference apres denoising", resize_to_fit(np.hstack((img1_denoised, cv2.convertScaleAbs(diff(img1, img1_denoised), alpha=3, beta=0)))))

# cv2.imshow("Deplacement entre img1 et img2", resize_to_fit(np.hstack((img1_denoised_color, cv2.addWeighted(img2_denoised_color, 0.6, draw_direction_wheel(motion_to_hsv(motion_images(img1_denoised, img2_denoised), img1_denoised), radius=80, position=(100, 100)), 0.4, 0)))))
# cv2.imshow("Deplacement entre img2 et img3", resize_to_fit(np.hstack((img2_denoised_color, cv2.addWeighted(img3_denoised_color, 0.6, draw_direction_wheel(motion_to_hsv(motion_images(img2_denoised, img3_denoised), img1_denoised), radius=80, position=(100, 100)), 0.4, 0)))))
# cv2.imshow("Deplacement entre img3 et img4", resize_to_fit(np.hstack((img3_denoised_color, cv2.addWeighted(img4_denoised_color, 0.6, draw_direction_wheel(motion_to_hsv(motion_images(img3_denoised, img4_denoised), img1_denoised), radius=80, position=(100, 100)), 0.4, 0)))))
# cv2.imshow("Deplacement entre img4 et img5", resize_to_fit(np.hstack((img4_denoised_color, cv2.addWeighted(img5_denoised_color, 0.6, draw_direction_wheel(motion_to_hsv(motion_images(img4_denoised, img5_denoised), img1_denoised), radius=80, position=(100, 100)), 0.4, 0)))))
# cv2.imshow("Deplacement entre img5 et img6", resize_to_fit(np.hstack((img5_denoised_color, cv2.addWeighted(img6_denoised_color, 0.6, draw_direction_wheel(motion_to_hsv(motion_images(img5_denoised, img6_denoised), img1_denoised), radius=80, position=(100, 100)), 0.4, 0)))))
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# prompt canal pour prendre les images d'entrées
# prendre en compte le repere pour les transfomrations geometriques via matrice
# afficher les axes
# afficher les deplacements via delta Z en temperature (comme sur le dashboard )
# representer la norme quadratique pour la 2D


def export_pngs(side):
    output_dir = f"./output_heatmap/{side}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 30):
        a = cv2.imread(f"./data/{side}/img{i}.tiff", cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f"./data/{side}/img{i+1}.tiff", cv2.IMREAD_GRAYSCALE)

        if a is None or b is None:
            print(f"⚠️ Image manquante à t={i}, sautée.")
            continue

        a_denoised = denoise(a)
        b_denoised = denoise(b)

        flow = motion_images(a_denoised, b_denoised)
        magnitude = magnitude_map(flow)

        # 📊 Affichage avec matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(magnitude, cmap="viridis", origin="upper")
        ax.set_title(f"Déplacement t{i} → t{i+1}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude du déplacement (pixels)")

        output_path = f"{output_dir}/img{i}-to-img{i+1}.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"{output_path} exporté.")

def export_animation(side):
    input_path = f'./output_heatmap/{side}'
    output_path = f'./output_heatmap/animations/{side}_transitions.mp4'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
    if not images:
        print("aucune image trouvée dans", input_path)
        return

    print(f"{len(images)} images trouvées.")

    first_frame = cv2.imread(os.path.join(input_path, images[0]))
    if first_frame is None:
        print("erreur: premiere frame == None")
        return

    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"image illisible : {img_name}")
            continue
        out.write(frame)

    out.release()
    print("animation des frames exportée avec succès :", output_path)


# export_pngs("right")
export_animation("right")
