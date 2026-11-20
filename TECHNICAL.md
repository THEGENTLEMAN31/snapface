# SnapFace — décisions techniques

Ce document consigne les choix d'implémentation de la v2, les mesures qui les
justifient, et les options écartées. Il est destiné à la relecture technique :
le README explique l'usage, ce fichier explique le pourquoi.

## Contrainte directrice

Une frame à 30 fps dispose d'un budget de 33 ms. Tout ce qui n'est pas
indispensable dans la boucle est mesuré, et le GPU n'est utilisé que là où il
apporte quelque chose : la grille d'échantillonnage du visage.

Mesures (RTX 3050 Laptop, crop 256, frame 1280×720, `scripts/bench.py`) :

| Étage | CUDA (médiane) | CPU (médiane) | Rôle |
|---|---|---|---|
| détection YuNet (480 px) | 6,89 ms | 8,09 ms | avant-plan, 1 frame sur 2 |
| copie vers tensor | 0,47 ms | 0,49 ms | CPU → GPU |
| warp `grid_sample` | **0,13 ms** | 1,46 ms | cœur de l'effet |
| retour en uint8 | 0,68 ms | 0,38 ms | GPU → CPU |
| fusion + feather | 2,71 ms | 3,02 ms | numpy |
| **total / frame** | **10,87 ms** | 13,43 ms | |
| débit théorique | 92 fps | 74 fps | |
| débit mesuré (bout en bout) | **76,8 fps** | 59,8 fps | visage détecté sur 100 % des frames |

Conséquence contre-intuitive et assumée : **le warp GPU ne représente que 1,2 %
du budget**. Le coût est la détection et la fusion, pas la déformation. C'est
précisément pour cela que PyTorch est cantonné au crop.

## Décisions

### 1. YuNet (ONNX) plutôt que MediaPipe FaceMesh

YuNet détecte et fournit 5 landmarks pour ~7 ms sur un frame de 480 px, sans
dépendance supplémentaire (`cv2.FaceDetectorYN` lit l'ONNX directement).
MediaPipe apporterait 468 points (déformations par zones du visage, morphing
plus crédible) mais ajoute une dépendance, un modèle à télécharger et un coût
d'inférence supérieur sur un CPU déjà chargé.

Décision : YuNet pour la v2. MediaPipe reste la piste d'extension pertinente si
l'objectif devient « morphing facial crédible » plutôt que « déformation
géométrique temps réel ».

### 2. Correctif : le format de sortie de `FaceDetectorYN` a changé

En OpenCV 5, chaque ligne vaut `[x, y, w, h, 5×(lx, ly), score]` : le score est
la **dernière** colonne, pas la cinquième. Le code initial lisait
`row[4]` comme score (on obtenait 239,3 au lieu de 0,92) et décalait les
landmarks d'un cran. Les effets centrés (bulge, swirl) le masquaient ; les
effets ancrés l'ont révélé immédiatement.

Régression couverte par `tests/test_detector.py` : score dans `]0,1]`,
landmarks dans la boîte, ordre anatomique yeux < nez < bouche.

### 3. Pas de rotation d'image : des ancres dans le repère du visage

Aligner le visage par rotation d'image impose un rééchantillonnage
aller-retour (rotation, effet, rotation inverse) : deux interpolations
supplémentaires, donc de la perte de netteté et du flou de bord.

Décision : le crop reste axis-aligned ; les 5 landmarks normalisés sont passés
aux effets, qui expriment leurs déformations dans le repère du visage
(`anchors_in_crop`). Les effets deviennent insensibles à la translation et à
l'échelle du visage, sans coût de rééchantillonnage.

Limite assumée : un effet *directionnel* (miroir selon l'axe du nez) suivrait
mieux une tête très inclinée avec une vraie dé-rotation. Aucun effet livré n'est
directionnel.

### 4. One Euro plutôt qu'une moyenne exponentielle

L'EMA sur `(x, y, w, h)` introduit un compromis jitter/latence non contrôlable.
Le filtre One Euro module sa coupure selon la vitesse estimée : peu de
latence en mouvement, beaucoup de lissage à l'arrêt.

Mesure A/B sur 200 frames de visage en mouvement (translation + rotation) :

| | sans lissage | One Euro | gain |
|---|---|---|---|
| instabilité du centre (px) | 5,73 | **2,82** | ÷ 2,0 |
| variation de la taille (CV) | 3,03 % | 2,40 % | −21 % |

L'EMA précédente n'était pas mesurée ; elle n'est plus dans le code.

### 5. Lucas-Kanade épars plutôt que flux dense

La détection tourne une frame sur deux (`--det-every`). Entre deux détections,
les 5 landmarks sont suivis par `calcOpticalFlowPyrLK`, avec contrôle
aller-retour (un point dont le retour s'écarte de plus de 1,5 px est rejeté) et
seuil de 3 points valides. Si le suivi échoue, la boîte est conservée puis
expire après 0,6 s.

Le flux **dense** (Farnebäck/DIS) a été écarté : 3 à 5 ms par frame pour un gain
marginal, et un blending temporel guidé par le flux introduit du ghosting dès
que l'estimation se trompe. L'usage 1 (suivi épars) corrige le défaut visible :
plus de zone déformée figée quand le détecteur décroche.

### 6. Détection découplée par échantillonnage temporel, pas par thread

Un thread de détection supprimerait le blocage, au prix d'une boîte aux lettres
partagée, de courses possibles sur `cv2.VideoCapture` et d'un comportement non
déterministe (donc intestable).

Décision : `--det-every 2` + suivi LK. Effet mesuré : le coût de détection est
amorti de moitié (6,89 ms → ~3,4 ms par frame en moyenne), la boucle reste
mono-thread et reproductible.

### 7. Fusion : masque elliptique, alpha prémultiplié, cache

Le code initial construisait un masque rempli de 1,0 puis flouté : le flou d'une
constante reste une constante, donc **le feather ne faisait rien**. La couture
était invisible uniquement parce que le warp était identité au bord.

Corrections :
- masque smoothstep réel, de forme **elliptique** (la forme d'un visage) : un
  masque rectangulaire rendait le kaleidoscope et le miroir « carrés » ;
- fusion par `out = roi + m·(face − roi)` en float32, sans allocation inutile ;
- masque mémorisé sur une taille quantifiée au multiple de 16 puis redimensionné
  (la bbox varie à chaque frame, un cache par taille exacte ne touchait jamais) :
  fusion 4,7 ms → **2,71 ms**.

Mesure de contrôle (`seam_error`, écart max sur l'anneau de bord) :
`identity` = 2 niveaux de gris ; `swirl` = 9 ; `mirror` et `kaleidoscope` = 10
(ces deux effets modifient tout le champ, la valeur mesure la déformation, pas
une couture).

### 8. Effets ancrés plutôt que centrés

`bulge`/`pinch`/`swirl` sont centrés sur le crop : le centre géométrique du
visage n'est pas un point d'intérêt anatomique. `big_eyes` et `big_mouth`
utilisent les landmarks : deux zones de gonflement centrées sur les yeux et la
bouche, avec un poids `(1 − (r/R)²)²` (plat au centre, doux au bord).

Le même poids est utilisé pour `bulge`/`pinch`, conformément à la formule
d'origine : un poids `(1 − r/R)²` concentrait la déformation sur quelques
pixels et la rendait à peine visible.

### 9. Transitions par interpolation de grilles

Deux effets de même taille produisent deux grilles `(1,2,H,W)`. Une seule
`grid_sample` sur une grille interpolée (`lerp`) donne une transition continue
entre deux effets, sans fondu d'image ni double calcul d'échantillonnage.
Coût : deux `warp()` au lieu d'un pendant 0,45 s, soit +0,13 ms.

### 10. Export vidéo : codec réel, cadence réelle

`avc1` échoue sur cette machine (pas de `h264_v4l2m2m`), `mp4v` fonctionne :
le writer sonde les codecs au lieu de supposer. La boucle tourne en temps réel
variable alors que le writer a une cadence fixe ; sans correction, une capture
de 12 s à 7 fps produisait un MP4 de 2,9 s. Le fichier est donc ré-encodé à la
fermeture avec ffmpeg, en H.264 CRF 18, à la **cadence mesurée** (`-r` avant
`-i`) : la durée du fichier correspond au temps réel.

### 11. Options écartées

- **Transfert de style neuronal (AdaIN)** : ~18 ms par frame sur ce GPU, plus
  une incohérence temporelle qui exigerait un blending par flux — pour un
  résultat visuellement déjà vu partout, et qui détournerait la thèse du projet.
  Si un « look » est souhaité, un LUT ou un rendu toon déterministe coûte
  0,5 ms et ne scintille pas par construction.
- **Réseau de flux optique (RAFT)** : 30 à 50 ms, hors budget.
- **Thread de détection**, voir §6.
- **`torch.compile` / fp16** : `grid_sample` est limité par la bande passante
  mémoire, pas par le calcul ; aucun gain attendu à 0,13 ms.

## Limites connues

- **Une seule personne** : `detect()` garde la détection la plus confiante.
  Le multi-visages demande un état de suivi par visage (structure prête :
  `BoxTracker` est instanciable plusieurs fois).
- **Pas de dé-rotation** : voir §3.
- **Caméra** : la qualité perçue dépend surtout de l'éclairage. Sur un
  contre-jour, le détecteur ne trouve rien et le pipeline rend la frame brute,
  ce qui est le comportement attendu mais peu spectaculaire.
- **`--seconds` en mode `video`** compte en temps vidéo, en mode webcam en temps
  réel.

## Reproductibilité

```bash
./.venv/bin/python -m pytest                      # 41 tests
./.venv/bin/python scripts/bench.py --json b.json # perfs + jitter A/B
./.venv/bin/python scripts/generate_samples.py --dir samples
./.venv/bin/python scripts/review.py --video out.mp4 --strict
```

Le banc d'essai a besoin d'une vidéo contenant un visage : `--video` accepte
n'importe quel fichier local, et le script `bench.py` retombe sur une photo fixe
si la vidéo est absente.
