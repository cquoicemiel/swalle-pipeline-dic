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
├── utils/
│   ├── config_loader.py
│   ├── deplacement.py
│   ├── encodage_video.py
│   ├── export.py
│   ├── logger.py
│   ├── pretraitement.py
│   ├── stereo.py
└── main.py
```


## Utilisation
Placer les photos issues des deux caméras ``/data/left`` pour les photos de la caméra principale et ``/data/right`` pour celles de la caméra secondaire.

Exécuter le fichier ``main.py`` en modifiant ou nom les instructions, par défaut le programme exporte toutes les transitions avec les déplacements représentés en heatmap pour chaque angle et aussi les videos des transitions associées, qui sont rien de plus que l'enchaînement des images. Les vidéos sont encodé en H.265, permettant leur intégration web et autres utilisations.
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