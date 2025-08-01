from utils.export import export_animation, export_pngs
from utils.encodage_video import encodage_mp4_web



export_pngs("left", "./output_heatmap")
export_pngs("right", "./output_heatmap")

export_animation("left", "./output_heatmap", "./output_heatmap/animations")
export_animation("right", "./output_heatmap", "./output_heatmap/animations")


encodage_mp4_web(
    input_path="./output_heatmap/animations/left_transitions.mp4",
    output_path="./output_heatmap/animations/left_transitions_web.mp4",
    resolution=None,
    framerate=30,
    crf=23,
    preset="fast",
)
encodage_mp4_web(
    input_path="./output_heatmap/animations/right_transitions.mp4",
    output_path="./output_heatmap/animations/right_transitions_web.mp4",
    resolution=None,
    framerate=30,
    crf=23, # + le paramètre est haut plus la compression est important
    preset="fast", # "slow" pour une meilleur compression mais en + de temps
)