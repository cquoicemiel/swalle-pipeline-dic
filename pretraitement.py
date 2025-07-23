import cv2
import numpy as np
import matplotlib.pyplot as plt

def denoise(img):
    img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
    return img_denoised


def img_dim(img): 
    return img.shape


def link_matches(a, b):
    # On relie les éléments qui match le mieux
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Afficher les meilleurs matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], None, flags=2) # en ajustant la limite du slice de matches on peut ajuster l'exigeance du matching
    return img_matches



def resize_to_fit(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # ne pas agrandir si déjà petit
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized




if __name__ == "__main__":

    # on charge les images
    img1 = cv2.imread("./data/right/img0.tiff", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./data/left/img0.tiff", cv2.IMREAD_GRAYSCALE)

    # on affiche les images
    diff = cv2.absdiff(img1, denoise(img1))
    cv2.imshow("Originale", resize_to_fit(img1))
    cv2.imshow("Filtrée (Bilateral)", resize_to_fit(denoise(img1)))
    cv2.imshow("Différence (Original - Filtrée)", resize_to_fit(diff))
    cv2.imshow("Elements liés", resize_to_fit(link_matches(denoise(img1), denoise(img2))))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


