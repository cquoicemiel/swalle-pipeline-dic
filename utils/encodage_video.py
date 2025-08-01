import ffmpeg
import os
from logger import get_logger

logger = get_logger(__name__)

def encodage_mp4_web(input_path, output_path, resolution=None, framerate=30, crf=23, preset='fast'):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    output_args = {
        'vcodec': 'libx264',
        'crf': crf, # + le paramètre est haut plus la compression est important    
        'preset': preset, # "slow" pour une meilleur compression mais en + de temps
        'r': framerate,
        'movflags': '+faststart',
        'pix_fmt': 'yuv420p',
    }


    output_args['an'] = None  # pas d'audio

    stream = ffmpeg.input(input_path)

    if resolution:
        width, height = resolution
        stream = stream.filter('scale', width, height)

    (
        stream
        .output(output_path, **output_args)
        .run(overwrite_output=True)
    )

    logger.info(f"vidéo encodée avec succès: {output_path}")
