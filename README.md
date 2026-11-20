# SnapFace

Déformation faciale **temps réel** en Python. Détection par **OpenCV + YuNet
(ONNX)**, effet calculé en **PyTorch** (`torch.nn.functional.grid_sample`),
affichage live et export vidéo H.264.

Le visage est détecté, suivi entre deux détections par **flux optique
Lucas-Kanade**, puis déformé par une **grille d'échantillonnage GPU** recollée
avec un masque de fusion elliptique. Le reste de la boucle (capture, fusion,
encodage) reste en OpenCV/numpy : PyTorch ne touche que le crop du visage.

Mesuré sur RTX 3050 Laptop, 1280×720, visage détecté sur 100 % des frames :
**76,8 fps** de bout en bout, dont **0,13 ms** pour le warp lui-même.

## Démarrage

```bash
cd /home/jose/projects/snap_face
./.venv/bin/python main.py            # webcam
```

Ou en installation classique :

```bash
pip install -e . && snapface
```

Touches en cours d'exécution :

| Touche | Action |
|---|---|
| `1` … `8` | changer d'effet (transition morph) |
| `d` | afficher/masquer le maillage de déformation |
| `s` | split avant/après |
| `q` / `Échap` | quitter |

## Effets

| Touche | Effet | Paramètre | Description |
|---|---|---|---|
| 1 | `identity` | — | témoin, image inchangée |
| 2 | `bulge` | `strength`, `radius` | gonflement radial |
| 3 | `pinch` | `strength`, `radius` | pincement radial |
| 4 | `swirl` | `swirl_strength` | tourbillon |
| 5 | `mirror` | — | symétrie horizontale |
| 6 | `kaleidoscope` | `segments`, `spin` | repli angulaire |
| 7 | `big_eyes` | `strength`, `radius` | yeux agrandis (ancré sur les landmarks) |
| 8 | `big_mouth` | `strength`, `radius` | bouche agrandie (ancré sur les landmarks) |

## Commandes utiles

```bash
# enregistrement propre, 10 s, sans overlay
./.venv/bin/python main.py --seconds 10 --record post.mp4

# démo : cycle d'effets toutes les 2 s, avec transitions et pulsation
./.venv/bin/python main.py --demo 2 --pulse 0.25 --seconds 12 --record demo.mp4

# plan technique : maillage de déformation visible dans la vidéo
./.venv/bin/python main.py --effect swirl --debug-grid --seconds 6 --record grid.mp4

# split avant/après
./.venv/bin/python main.py --effect big_eyes --split --record split.mp4

# image seule (sans webcam)
./.venv/bin/python main.py --source image --image photo.jpg --effect bulge

# vidéo en entrée
./.venv/bin/python main.py --source video --video in.mp4 --record out.mp4

# effet et paramètres
./.venv/bin/python main.py --effect bulge --effect-param strength=0.8
./.venv/bin/python main.py --effect kaleidoscope --effect-param segments=8

# contrôle du suivi (détection une frame sur N)
./.venv/bin/python main.py --det-every 3 --min-cutoff 1.5 --beta 0.08
```

Liste complète : `./.venv/bin/python main.py --help`.

`--record` active automatiquement `--clean` : la vidéo ne contient jamais
l'overlay texte/FPS, seulement l'image traitée (et éventuellement le maillage si
`--debug-grid` est demandé).

## Vérification

```bash
./.venv/bin/python -m pytest                        # 41 tests
./.venv/bin/python scripts/generate_samples.py      # PNG par effet + index.html
./.venv/bin/python scripts/bench.py                 # ms/étage, GPU vs CPU, jitter
./.venv/bin/python scripts/review.py --video out.mp4 --strict
```

`scripts/review.py` envoie une copie légère de la vidéo à un modèle vision
(Gemini) qui rend un verdict JSON : visage détecté, déformation lisible, zone
stable/dérive/saute, couture visible. C'est un **second avis** sur ce que les
chiffres ne captent pas ; les mesures restent le juge principal.

## Notes

- **Matériel** : testé sur NVIDIA RTX 3050 Laptop (4 Go) / CUDA 13.0,
  `torch.cuda.is_available()` → `True`. Le CPU fonctionne aussi (59,8 fps).
- **Wayland/Hyprland** : le backend Qt est forcé sur `xcb` (XWayland), OpenCV 5
  n'embarquant pas de plugin wayland.
- **Codec** : `avc1` n'est pas disponible ici (pas d'encodeur V4L2) ; le writer
  sonde `avc1` puis `mp4v`, puis ré-encode en H.264 via ffmpeg à la cadence
  réelle mesurée. La durée du fichier correspond donc au temps réel.
- **Headless** : sans `DISPLAY`, le programme passe automatiquement en mode sans
  fenêtre ; combiner `--clean --record` pour un usage sans écran.
- **Éclairage** : la caméra doit voir un visage de face et éclairé. En contre-
  jour, aucune détection n'a lieu et la frame brute est rendue.

## Documentation

- [`TECHNICAL.md`](TECHNICAL.md) : décisions d'implémentation, mesures, limites.
