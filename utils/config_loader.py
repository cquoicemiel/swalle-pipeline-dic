import configparser
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

cfg_path = "./data/config/config_cam.cfg"

def load_config(path=cfg_path):
    logger.info(f"Chargement du fichier {path}.cfg.")
    config = configparser.ConfigParser()
    config.read(path)
    logger.info("Fichier config chargé.")
    return config

def load_stereo_settings(path=cfg_path):

    config = load_config(path)

    # cam principale
    logger.info(f"récupération des paramètres de la caméra principale dans le fichier {path}.cfg...")
    fc1_L = float(config["cam_principale"]["fc1"])
    fc2_L = float(config["cam_principale"]["fc2"])
    cc1_L = float(config["cam_principale"]["cc1"])
    cc2_L = float(config["cam_principale"]["cc2"])
    kc1_L = float(config["cam_principale"]["kc1"])
    kc2_L = float(config["cam_principale"]["kc2"])
    kc3_L = float(config["cam_principale"]["kc3"])
    kc4_L = float(config["cam_principale"]["kc4"])
    kc5_L = float(config["cam_principale"]["kc5"])

    K_left = np.array([[fc1_L, 0, cc1_L],
                       [0, fc2_L, cc2_L],
                       [0, 0, 1]])
    D_left = np.array([kc1_L, kc2_L, kc3_L, kc4_L, kc5_L])

    # cam secondaire
    logger.info("caméra secondaire...")
    fc1_R = float(config["cam_secondaire"]["fc1"])
    fc2_R = float(config["cam_secondaire"]["fc2"])
    cc1_R = float(config["cam_secondaire"]["cc1"])
    cc2_R = float(config["cam_secondaire"]["cc2"])
    kc1_R = float(config["cam_secondaire"]["kc1"])
    kc2_R = float(config["cam_secondaire"]["kc2"])
    kc3_R = float(config["cam_secondaire"]["kc3"])
    kc4_R = float(config["cam_secondaire"]["kc4"])
    kc5_R = float(config["cam_secondaire"]["kc5"])

    K_right = np.array([[fc1_R, 0, cc1_R],
                        [0, fc2_R, cc2_R],
                        [0, 0, 1]])
    D_right = np.array([kc1_R, kc2_R, kc3_R, kc4_R, kc5_R])

    # matrice de rotation
    logger.info("composition de la matrice de rotation...")
    R = np.array([
        [float(config["calib_stereo"]["Rs11_12"]), float(config["calib_stereo"]["Rs12_12"]), float(config["calib_stereo"]["Rs13_12"])],
        [float(config["calib_stereo"]["Rs21_12"]), float(config["calib_stereo"]["Rs22_12"]), float(config["calib_stereo"]["Rs23_12"])],
        [float(config["calib_stereo"]["Rs31_12"]), float(config["calib_stereo"]["Rs32_12"]), float(config["calib_stereo"]["Rs33_12"])]
    ])

    # matrice de translation
    logger.info("composition de la matrice de translation...")
    T = np.array([
        float(config["calib_stereo"]["Ts1_12"]),
        float(config["calib_stereo"]["Ts2_12"]),
        float(config["calib_stereo"]["Ts3_12"])
    ]) / 1000.0

    logger.info("paramètres extraits et matrice calculées.")
    return K_left, D_left, K_right, D_right, R, T