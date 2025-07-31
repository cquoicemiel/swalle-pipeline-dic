import ffmpeg
import os

def encodage_mp4_web(input_path, output_path, resolution=None, framerate=30, crf=23, preset='fast', keep_audio=False):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    output_args = {
        'vcodec': 'libx264',
        'crf': crf,
        'preset': preset,
        'r': framerate,
        'movflags': '+faststart',
        'pix_fmt': 'yuv420p',
    }

    if not keep_audio:
        output_args['an'] = None  # no audio

    stream = ffmpeg.input(input_path)

    if resolution:
        width, height = resolution
        stream = stream.filter('scale', width, height)

    (
        stream
        .output(output_path, **output_args)
        .run(overwrite_output=True)
    )

    print(f"vidéo encodée avec succès: {output_path}")
