# SnapFace

Filtres faciaux **temps réel** en Python : détection de visage par **OpenCV + YuNet (ONNX)** et effets de déformation géométrique calculés en **PyTorch** (`torch.nn.functional.grid_sample`). Conçu pour produire un rendu propre, enregistrable en MP4 pour un post LinkedIn.

## Démarrage (environnement déjà prêt)

```bash
cd /home/jose/projects/snap_face
./.venv/bin/python main.py
```

Webcam par défaut (indice 0). Touches :
- `1`..`6` : changer d'effet
- `q` ou `Échap` : quitter

## Utilisation

```bash
# Webcam live avec affichage (nom d'effet + FPS en haut à gauche)
./.venv/bin/python main.py

# Clean : pas d'overlay, + enregistrement MP4 pour LinkedIn
./.venv/bin/python main.py --clean --record post.mp4

# Mode démo : cycle automatique d'effets toutes les 5 secondes
./.venv/bin/python main.py --demo 5 --clean --record post.mp4

# Image seule (sans webcam)
./.venv/bin/python main.py --source image --image photo.jpg --effect bulge

# Vidéo en entrée
./.venv/bin/python main.py --source video --video input.mp4 --record out.mp4

# Choisir effet + paramètres (ex: gonflement fort, tourbillon doux)
./.venv/bin/python main.py --effect bulge --effect-param strength=0.8
./.venv/bin/python main.py --effect swirl --effect-param swirl_strength=2.5

# Périphérique PyTorch : auto (recommandé), cuda ou cpu
./.venv/bin/python main.py --device auto
```

Détails : `./.venv/bin/python main.py --help`

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

Génère `samples/identity_<ts>.png` … `samples/kaleidoscope_<ts>.png` + `index.html`, avec assertions de non-régression (forme/domaine/identité).

## Notes

- **Matériel de référence** : utilisé avec NVIDIA RTX 3050 Laptop (4 Go VRAM) / CUDA — `torch.cuda.is_available()` → `True`.
- **Headless (pas d'écran)** : l'affichage nécessite un `DISPLAY`/`WAYLAND_DISPLAY`. En mode headless, utiliser `--clean` (+ `--record`) : aucun overlay, aucune fenêtre ; le programme bascule automatiquement en mode sans affichage si aucun display n'est disponible.
- Enregistrement : codec `avc1` (H.264) puis repli `mp4v` ; le flux écrit est toujours la version propre (sans overlay), quel que soit `--clean`.
- Enregistrement automatiquement propre : `--record` active `--clean`.
- Date : *current year — 2026*.