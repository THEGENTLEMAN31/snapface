# SnapFace

Filtres faciaux **temps réel** en Python. Détection de visage par **OpenCV + YuNet (ONNX)**, effets de déformation géométrique calculés en **PyTorch** (`torch.nn.functional.grid_sample`), affichage live et enregistrement MP4.

## Démarrage

Environnement virtuel déjà prêt (Python 3.14, torch, opencv-python, numpy).

```bash
cd /home/jose/projects/snap_face
./.venv/bin/python main.py
```

Ouvre ta webcam (indice 0) dans une fenêtre « SnapFace ». Touches :

- `1`..`6` : changer d'effet
- `q` ou `Échap` : quitter

## Utilisation

```bash
# Webcam live avec affichage (nom d'effet + FPS en haut à gauche)
./.venv/bin/python main.py

# Enregistrement MP4 (propre, sans overlay)
./.venv/bin/python main.py --clean --record post.mp4

# Mode démo : cycle automatique d'effets toutes les 5 secondes
./.venv/bin/python main.py --demo 5 --clean --record post.mp4

# Image seule (sans webcam)
./.venv/bin/python main.py --source image --image photo.jpg --effect bulge

# Vidéo en entrée
./.venv/bin/python main.py --source video --video input.mp4 --record out.mp4

# Choisir un effet et ses paramètres
./.venv/bin/python main.py --effect bulge --effect-param strength=0.8
./.venv/bin/python main.py --effect swirl --effect-param swirl_strength=2.5

# Périphérique PyTorch : auto (recommandé), cuda ou cpu
./.venv/bin/python main.py --device auto
```

Liste complète des options : `./.venv/bin/python main.py --help`

## Effets

| Touche | Effet | Paramètre clé |
|--------|-------|---------------|
| 1 | identity (aucun) | — |
| 2 | bulge (gonflement) | `strength` |
| 3 | pinch (pincement) | `strength` |
| 4 | swirl (tourbillon) | `swirl_strength` |
| 5 | mirror (miroir) | — |
| 6 | kaleidoscope | `segments` |

## Vérification sans webcam

```bash
./.venv/bin/python scripts/generate_samples.py --dir samples --device cuda
```

Génère `samples/identity_<ts>.png` … `samples/kaleidoscope_<ts>.png` plus un `index.html` pour inspection visuelle, avec assertions de non-régression (formes, bornes [0,1], identité bit-à-bit de l'effet `identity`).

## Notes

- **Matériel** : testé avec un NVIDIA RTX 3050 Laptop (4 Go VRAM) / CUDA — `torch.cuda.is_available()` → `True`.
- **Wayland/Hyprland** : le backend d'affichage Qt est forcé sur `xcb` (via XWayland), car OpenCV 5.0 n'embarque pas de plugin wayland.
- **Enregistrement** : codec `avc1` (H.264) avec repli automatique sur `mp4v`. Le flux écrit est toujours la version propre (sans overlay), quel que soit `--clean` ; `--record` active d'ailleurs `--clean`.
- **Headless (sans écran)** : l'affichage nécessite un `DISPLAY`/`WAYLAND_DISPLAY`. Sans display, le programme bascule automatiquement en mode sans fenêtre — combiner avec `--clean --record` pour un usage entièrement sans écran.
- **Performance** : détection accélérée en réduisant le frame à 480 px de large avant l'inférence YuNet (coordonnées rescallées), effet GPU sur crop 256 px — boucle complète ≈ 45 images/s en 1280×720 sur le poste de référence.