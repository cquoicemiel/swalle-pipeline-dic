# Pipeline de post-traitement DIC | Swall-E

Pipeline **Python** qui prend en entrée des photos aux format .tiff issues de prises de photos par deux caméras *synchronisées* et *calibrées* ainsi que leur fichier de calibration au format .cfg puis applique différents traitements d'images sur ces photos et notamment renvoie en sortie les déplacements illustrés en 2D et en 3D.
## Arborescence 
Voici la structure du projet:
```
swalle-pipeline-dic/
├── data/
│   ├── left/
│   └── right/
├── output_heatmap/
│   ├── animations/
│   ├── left/
│   └── right/
├── deplacement.py
├── encodage_video.py
├── export.py
├── pretraitement.py
├── stereo.py
└── main.py
```


## Utilisation
Placer les photos dans ``/data/left`` et ``/data/right`` chaque angle correspondant à une des deux caméras utilisées 
### Fonctions

## Modules utilisés
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/stable/)
- [Numpy](https://numpy.org/)
- [Three.js](https://threejs.org/)
- [ffmpeg](https://ffmpeg.org/)
## Accès local
Si vous avez accès au repository GitHub, vous pouvez cloner ce dernier, installer les dépendances puis héberger le projet sur votre machine.

Clonage du repository:
```bash
git clone https://github.com/cquoicemiel/dashboard-swalle
cd swalle-pipeline-dic
```
Installation des dépendances:
```bash
pip install -r requirements.txt
```
