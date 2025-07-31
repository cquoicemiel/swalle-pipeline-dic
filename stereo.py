import cv2
import numpy as np
import os
import open3d as o3d

# CONFIG CORRIGÉE
# Caméra principale (gauche)
K_left = np.array([[10731.6, 0, 1257.89],
                   [0, 10731.8, 1010.97],
                   [0, 0, 1]])

# Caméra secondaire (droite)  
K_right = np.array([[10776.4, 0, 1223.65],
                    [0, 10781, 997.056],
                    [0, 0, 1]])

# Correction des coefficients de distortion (format OpenCV: [k1, k2, p1, p2, k3])
# Conversion depuis kc1,kc2,kc3,kc4,kc5 vers k1,k2,p1,p2,k3
D_left = np.array([-0.0449469, 1.2968, 0.000686857, 0.000534226, 0])
D_right = np.array([-0.0388554, 0.702762, 0, 0, 0])

# Matrice de rotation corrigée (transpose pour inverser gauche↔droite)
R_orig = np.array([[0.863015, -0.00155128, -0.505176],
                   [0.000800376, 0.999998, -0.00170345],
                   [0.505178, 0.00106577, 0.863014]])
R = R_orig.T  # Transpose pour corriger l'ordre

# Translation corrigée (conversion en mètres + inversion si nécessaire)
T_orig = np.array([307.603, 0.932237, 77.9891]) / 1000.0
T = -R @ T_orig  # Correction de la translation

def process_pair(left_path, right_path, output_path, debug=False):
    # Chargement des images
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if imgL is None or imgR is None:
        print(f"impossible de charger les images: {left_path}, {right_path}")
        return False
        
    h, w = imgL.shape
    print(f"taille des images: {w}x{h}")

    # Rectification stéréo
    try:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, D_left, K_right, D_right,
            (w, h), R, T, 
            flags=cv2.CALIB_ZERO_DISPARITY, 
            alpha=0  # 0 = crop, 1 = keep all pixels
        )
        
        # Calcul des cartes de rectification
        map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, (w, h), cv2.CV_32FC1)

        # Application de la rectification
        rect_left = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
        
        if debug:
            cv2.imwrite(f"debug_rect_left_{os.path.basename(left_path)}", rect_left)
            cv2.imwrite(f"debug_rect_right_{os.path.basename(right_path)}", rect_right)
        
    except Exception as e:
        print(f"erreur lors de la rectification: {e}")
        return False

    # Paramètres de disparité adaptés aux hautes résolutions
    # Calcul estimation: baseline ≈ 308mm, focale ≈ 10700px
    # Pour un objet à 1m: disparité ≈ 308*10700/1000 ≈ 3296 pixels !
    
    # Paramètres corrigés pour votre configuration
    min_disp = 0
    max_disp = 512  # Augmenté significativement
    num_disp = max_disp - min_disp
    
    # S'assurer que num_disp est divisible par 16
    num_disp = ((num_disp + 15) // 16) * 16
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=11,  # Augmenté pour plus de robustesse
        P1=8 * 3 * 11**2,  # Ajusté selon blockSize
        P2=32 * 3 * 11**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,  # Réduit pour plus de permissivité
        speckleWindowSize=150,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print(f"calcul disparité (min={min_disp}, max={min_disp + num_disp})...")
    disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0
    
    # Statistiques sur la disparité
    valid_disp = disparity[disparity > 0]
    if len(valid_disp) > 0:
        print(f"disparité: min={valid_disp.min():.1f}, max={valid_disp.max():.1f}, "
              f"moyenne={valid_disp.mean():.1f}, points valides={len(valid_disp)}")
    else:
        print("aucune disparité valide trouvée!")
        
        if debug:
            # Sauvegarde pour debug
            disp_viz = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(f"debug_disparity_{os.path.basename(left_path)}", disp_viz)
        
        return False

    # Filtrage et reconstruction 3D
    disparity[disparity <= 0] = 0.1
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    
    # Masque plus permissif
    mask = (disparity > 0.1) & (disparity < num_disp) & np.isfinite(points_3D).all(axis=2)
    
    # Filtrage des points aberrants (optionnel)
    z_coords = points_3D[:,:,2]
    z_valid = (z_coords > 0.1) & (z_coords < 10.0)  # Entre 10cm et 10m
    mask = mask & z_valid

    points = points_3D[mask]
    
    if len(points) < 100:
        print(f"pas assez de points 3D valides: {len(points)}")
        return False

    # Chargement de l'image couleur pour les couleurs des points
    img_color = cv2.imread(left_path)
    if img_color is not None:
        img_color_rect = cv2.remap(img_color, map1x, map1y, cv2.INTER_LINEAR)
        colors = img_color_rect[mask] / 255.0
    else:
        colors = np.ones((len(points), 3)) * 0.5  # Gris par défaut

    # Export du nuage de points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"nuage exporté: {output_path} ({len(points)} points)")
    
    if debug:
        # Visualisation de la disparité
        disp_viz = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(f"debug_disparity_{os.path.basename(left_path)}", disp_viz)
    
    return True

# Test sur une paire pour debug
def test_single_pair():
    input_dir = "./data/" 
    output_dir = "./output_ply"
    
    # Test sur la première paire disponible
    for i in range(100):
        lpath = os.path.join(input_dir, f"left/img{i}.jpg")
        rpath = os.path.join(input_dir, f"right/img{i}.jpg")
        opath = os.path.join(output_dir, f"cloud_img{i}.ply")
        
        if os.path.exists(lpath) and os.path.exists(rpath):
            print(f"test sur la paire {i}")
            success = process_pair(lpath, rpath, opath, debug=True)
            if success:
                print("test réussi!")
            break
    else:
        print("aucune paire d'images trouvée dans ./data/left/ et ./data/right/")

# BOUCLE PRINCIPALE
def process_all_pairs():
    input_dir = "./data/" 
    output_dir = "./output_ply"
    
    success_count = 0
    total_count = 0
    
    for i in range(100):
        lpath = os.path.join(input_dir, f"left/img{i}.jpg")
        rpath = os.path.join(input_dir, f"right/img{i}.jpg")
        opath = os.path.join(output_dir, f"cloud_img{i}.ply")

        if os.path.exists(lpath) and os.path.exists(rpath):
            total_count += 1
            if process_pair(lpath, rpath, opath):
                success_count += 1
    
    print(f"\n résultats: {success_count}/{total_count} paires traitées avec succès")

if __name__ == "__main__":
    print("démarrage du traitement stéréo...")
    
    # test_single_pair()
    
    process_all_pairs()