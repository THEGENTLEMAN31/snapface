# AGENTS.md — SnapFace (spec destinée à l'agent de développement)

Tu es chargé d'implémenter **entièrement** le projet SnapFace. Ce document est ta référence unique : suis-le **dans l'ordre des phases (section 10)**, sans improviser l'architecture. L'environnement, les dépendances et le modèle de détection sont **déjà prêts** (cf. section 2) — tu n'as pas à les refaire.

> Règle d'or : chaque phase est terminée quand la **sortie de la phase 10.6 (tests d'acceptation)** est verte OU que tu as documenté dans `VERIFICATION.md` la raison exacte d'un échec avec la sortie console. Ne passe jamais à la phase suivante si la phase courante échoue sans `VERIFICATION.md`.

---

## 1. Objectif produit

Application Python **temps réel** de filtres faciaux :

- Capture webcam / image / vidéo.
- Détection de visage par **OpenCV + YuNet (ONNX)**.
- Application d'un effet de **déformation géométrique** calculé en **PyTorch** (`torch.nn.functional.grid_sample`).
- Affichage live avec changement d'effet à chaud (touches `1`-`6`), vitesse FPS, et **enregistrement MP4** du flux traité pour un post LinkedIn.

Contraintes non négociables :
1. **Propreté visuelle** : pas de flou de bord, pas de jitter de la boîte de visage (lissage temporel obligatoire), pas de module de texte à l'écran en mode `--clean`.
2. **boucle temps réel** : le crop du visage est le seul endroit où tourne PyTorch ; le reste reste en OpenCV/numpy.
3. **PyTorch doit réellement servir à l'effet** (pas un simple remplacement de cv2.remap par autre chose) : les grilles d'échantillonnage sont des tensors CUDA/CPU et passent par `grid_sample`.

---

## 2. Environnement (DÉJÀ PRÉPARÉ — ne rien réinstaller)

- OS : Linux (Debian, arch x86_64).
- Python : **3.14.7** (`/usr/bin/python3.14`).
- **Environnement virtuel : `/home/jose/projects/snap_face/.venv`** → tout lancement se fait via `./.venv/bin/python` (alias : définis `PYTHON=./.venv/bin/python`).
- Dépendances déjà installées : `torch 2.14.0` (wheels cp314, CUDA embarqué), `opencv-python`, `numpy`.
- **Modèle déjà téléchargé** : `assets/face_detection_yunet_2023mar.onnx` (232 Ko, objet LFS récupéré). Tu y accèdes tel quel, ne le retélécharge pas.
- **Webcam présente** : `/dev/video0` (indice caméra 0). Il y a aussi `/dev/video1`.
- **GPU détecté** : NVIDIA RTX 3050 Laptop (4 Go VRAM), CUDA 13.3. `torch.cuda.is_available()` doit renvoyer `True` (à vérifier : see section 10.6).
- Répertoire de travail : `/home/jose/projects/snap_face` (dépôt git initialisé, branche `main`).

Commande universelle pour exécuter le programme :
```bash
./.venv/bin/python main.py
```

---

## 3. Arborescence cible

```
snap_face/
├── main.py                  # point d'entrée CLI (webcam/image/video) — ne contient QUE du parsing et l'appel au Pipeline
├── snapface/
│   ├── __init__.py          # vide ou version
│   ├── config.py            # constantes + namespace Config (dmins/max de bbox, tailles, chemins)
│   ├── detector.py          # classe FaceDetector (YuNet) -> Detection
│   ├── geometry.py          # cv2<->torch, math des crops/bboxes, masks de fusion
│   ├── pipeline.py          # classe Pipeline : boucle complète capture->effet->affichage/enregistrement
│   └── effects/
│       ├── __init__.py      # registre d'effets + fonction get_effect(name, device)
│       ├── base.py          # classe abstraite Effect + dataclass EffectParams/EffetResult
│       ├── identity.py      # effet n°1 : aucun changement (témoin)
│       ├── bulge.py         # effet n°2 : gonflement radial
│       ├── pinch.py         # effet n°3 : pincement radial
│       ├── swirl.py         # effet n°4 : tourbillon
│       ├── mirror.py        # effet n°5 : miroir (symétrie)
│       └── kaleidoscope.py  # effet n°6 : kaléidoscope
├── scripts/
│   ├── generate_samples.py  # TEST HEADLESS : rend des PNG avant/après pour chaque effet (sans webcam)
├── assets/
│   └── face_detection_yunet_2023mar.onnx   # DÉJÀ PRÉSENT
├── requirements.txt
├── AGENTS.md                # ce fichier
```

---

## 4. Spécification des signatures (doivent être respectées à la lettre)

### 4.1 `snapface/config.py`
```python
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
YUNET_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"

@dataclass
class Config:
    cam_index: int = 0
    input_w: int = 1280              # largeur d'acquisition demandée
    input_h: int = 720
    crop_size: int = 256             # résolution (hauteur) du crop du visage envoyé à PyTorch
    effect: str = "identity"
    device: str = "auto"             # "auto" | "cuda" | "cpu"
    demo_seconds: float | None = None  # mode démo : cycle auto d'effets (None = off)
    clean: bool = False             # True = pas d'overlay texte/FPS à l'écran
    record: Path | None = None      # chemin .mp4 de sortie (None = pas d'enregistrement)
    smooth_alpha: float = 0.45      # lissage temporel de la bbox [0,1] — 1 = pas de lissage
    conf_threshold: float = 0.6
    # paramètres par défaut des effets (modifiables via --effect-param)
    effect_params: dict = field(default_factory=dict)
    preview: bool = False            # True = on n'applique PAS l'effet (affichage brut, debug)
    fps_target: int = 30
```

### 4.2 `snapface/geometry.py`
```python
def to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor
    # bgr: (H, W, 3) uint8 -> tensor (1, 3, H, W) float32, valeurs /255, on DEVICE

def from_tensor(t: torch.Tensor) -> np.ndarray
    # tensor (1, 3, H, W) -> numpy (H, W, 3) uint8, clip 0..255, chan dernier

def crop_face(frame: np.ndarray, det: Detection, expand: float = 0.35) -> tuple[np.ndarray, tuple]
    # découpe la région du visage avec marges autour du bbox + retourne la boundingbox
    # ajustée dans l'image d'origine (x, y, w, h). expand est la fraction de marge
    # (ex: 0.35 → 35% de la largeur/hauteur du visage en marge de chaque côté).

def paste_face(frame, face_img, box, feather: int = 15) -> np.ndarray
    # recolle le visage traité (redimensionné à la taille de `box`) dans le frame
    # avec un masque de fusion gaussien (feather en pixels) pour éviter la couture.

def soft_mask(h: int, w: int, feather_frac: float = 0.35, device="cpu") -> torch.Tensor
    # masque (1,1,H,W) float32 [0,1] : 1 au centre, 0 sur >=~feather_frac*min(h,w) des bords.
    # Transition douce (cosinus ou gaussienne). Servira à la fusion propre.

def clamp_bbox(box, frame_shape) -> tuple
```

### 4.3 `snapface/detector.py`
```python
@dataclass
class Detection:
    box: ...        # (x, y, w, h) en pixels dans le frame
    score: float
    landmarks: ...  # Optional[list[(x,y) x5]] — rempli si YuNet les fournit, sinon None

class FaceDetector:
    def __init__(self, conf_threshold: float = 0.6): ...
    def detect(self, frame_bgr: np.ndarray) -> Detection | None
        # YuNet via cv2.FaceDetectorYN sur YUNET_PATH.
        # Retourne la détection de plus haute confiance, sinon None.
    @property
    def using(self) -> str   # "yunet" | "none"
```

Détails YuNet (indispensables — **OpenCV 5.0** est installé, DONC ne PAS utiliser `cv2.dnn.readNetFromONNX` / `setInputSize` ni `cv2.CascadeClassifier`, ces API ont disparu) :
- **API officielle** : `fd = cv2.FaceDetectorYN.create(model=str(YUNET_PATH), config="", input_size=(320, 320), score_threshold=config.conf_threshold, nms_threshold=0.3, top_k=5000)`.
- Le détecteur exige que l'entrée match une taille : appeler `fd.setInputSize((w, h))` avec les dimensions du frame passé (ou redimensionner le frame d'abord).
- Inférence : `rows, dets = fd.detect(frame_bgr)` → `dets` est un array `(N, 15)` **en coordonnées du frame** : `[x, y, w, h, score, lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4]` ; `dets is None` si aucune face, `rows` = nombre de faces.
- Garder la détection au `score` max parmi `dets` (> score_threshold, déjà appliqué en interne).
- **Pas de fallback Haar** (API supprimée en cv2 5.0) : si YuNet est absent ou lève une erreur au `create`, le détecteur retourne `None` et `using` vaut `"none"` — le pipeline continue sans crop (frame brut). Vérifié : fonctionne sur le poste avec `input_size=(320,320)`.
- Performance : l'input 320x320 marche très bien ; tu peux aussi redimensionner le frame à ~480px de large puis `setInputSize` correspondant pour accélérer sous CPU.

### 4.4 `snapface/effects/base.py`
```python
class Effect(ABC):
    name: str = "base"
    def __init__(self, device: str = "cpu", **params): 
        self.device = device
        self.params = {...defaults...} | params        # params utilisateur fusionnés
    @abstractmethod
    def warp(self, grid: torch.Tensor) -> torch.Tensor:
        """Modifie la grille d'échantillonnage normalisée (n,2,H,W) -> (n,2,H,W)."""
    def apply(self, crop: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) crop tensor -> (1,3,H,W) warped. Implémentation à écrire:"""
        # 1. g = create_grid(H, W, device)   (grille normalisée [-1,1])
        # 2. g = self.warp(g)
        # 3. warped = F.grid_sample(crop, g, mode='bilinear', padding_mode='border', align_corners=True)
        # 4. retourner warped
```

Critère d'acceptation commun : l'application de l'effet ne change **jamais** les dimensions, les valeurs restent dans `[0,1]`, et `identity` renvoie une image identique (≈ erreur max < 1e-4).

### 4.5 Les 6 effets (formules EXACTES)

Convention : grille normalisée avec `u,v ∈ [-1,1]` (u = colonne, v = ligne), centre en `(0,0)`, **coin haut-gauche conventionnel involontaire** : utilise `align_corners=True` partout.

- **identity.py** : `warp` renvoie `grid` inchangé.
- **bulge.py** : rayon `R = 0.9` (en unité normalisée). Pour chaque pixel, `r = sqrt(u²+v²)`, si `r >= R` → rien. Sinon `factor = (1 - (r/R)**2)**2` (doux) ; le point est repoussé du centre : `u' = u * (1 - strength*factor)`, idem `v'`. `strength ∈ [0, 0.9]`, défaut `0.35`.
- **pinch.py** : pareil que bulge mais le point est attiré : `u' = u * (1 + strength*factor)` (inversion du signe), `strength` défaut `0.4`.
- **swirl.py** : rotation suivant `θ'(r) = θ + swirl_strength * (1 - r/R)` radian, `R=1.0`, `swirl_strength` défaut `1.4`. Implémentation via atan2/cos/sin de la grille.
- **mirror.py** : `u' = -abs(u)` (les deux côtés reflètent le côté droite du visage… plus simple : `u' = u` si `u < 0` sinon `u' = -u`). Applique aussi une **fusion douce sur 8% du centre** (fondu vertical) pour éviter la ligne dure si tu préfères ; plus simple : on prend `u' = u` pour `u<=0` et `u'=-u` pour `u>0` et on s'en tient là.
- **kaleidoscope.py** : paramètre `segments` (int, défaut 6). Convertit la grille en polaire `(r, θ)`. Plie l'angle : `sector = π / segments` ; `θ_folded = |(θ mod (2*sector)) - sector|` — attention aux signes : à chaque secteur, `θ_folded = abs(θ mod (2π/segments) - (π/segments))`. Reconstruit `(u', v')` avec `r` identique. (Si tu préfères `θ_folded = (π/segments) - abs(θ mod (2π/segments) - (π/segments))` pour un miroir correct, teste les deux et garde celui qui ne casse pas le centre en singularité navrée : assure-toi que `(0,0)` reste bien dans le domaine.)

**Aucun effondrement de performance** : ces opérations sont du vectoriel PyTorch, éventuellement pré-calculées pour le standard défaut (grid prête à l'emploi une fois par taille — met en cache la grille et éventuellement le `r/θ` une seule fois par effet).

### 4.6 `snapface/effects/__init__.py`
```python
_EFFECTS = {}  # nom -> classe
def register(cls): ...   # décorateur : enregistre cls.name
def get_effect(name: str, device: str = "cpu", params: dict | None = None) -> Effect
def available_effects() -> list[str]
```
Enregistre les 6 effets. `get_effect` lève `ValueError` si inconnu. `available_effects()` → `["identity","bulge","pinch","swirl","mirror","kaleidoscope"]`.

### 4.7 `snapface/pipeline.py`
```python
class Pipeline:
    def __init__(self, cfg: Config): ...
    def run(self) -> None: ...
```
Comportement :
1. `cv2.VideoCapture(cfg.cam_index)`, régler `CAP_PROP_FRAME_WIDTH/HEIGHT` (1280x720 par défaut; si échec, baisser à 960x540 puis 640x480 pour garder une boucle > ~25 FPS).
2. Boucle : `ret, frame = cap.read()` → `frame = cv2.resize(frame, (input_w, input_h))` si nécessaire.
3. `det = detector.detect(frame)`. Lissage temporel de la bbox : si `det` manque, bbox précédente conservée (avec expiration ~0.5 s) ; sinon `new_box = old_box*(1-alpha) + new_box*alpha` (per-pixel, sur les 4 valeurs), `alpha = cfg.smooth_alpha`.
4. `crop, box = crop_face(frame, det_smoothed, expand=0.35)` → resize à `crop_size` → `to_tensor` → `effect.apply(tensor)` → `from_tensor`. Si détection absente depuis trop longtemps, on affiche/écrit le frame brut (pas de crop).
5. Fusion : `paste_face(frame, face_img, box, feather=15)`.
6. Affichage : `cv2.imshow("SnapFace", frame)` avec overlay si `not cfg.clean` (nom de l'effet + FPS en haut à gauche via `cv2.putText`). `cv2.waitKey(1)` pour les touches : `'1'..'6'` → change l'effet (cf carte), `'q'`/`ESC` → quit. En mode `demo_seconds` : bascule auto d'effet toutes les `demo_seconds` secondes (cycle).
7. Enregistrement si `cfg.record` : `cv2.VideoWriter(str(cfg.record), fourcc, fps, (w,h))` — choisir fourcc en priorité `avc1` (H.264), sinon `mp4v`, sinon échouer proprement avec un message. **`frame` doit être écrit APRÈS l'affichage overlay mais avec `clean=True` recommandé** : en mode record, toujours écrire la version la plus propre (sans overlay) si `--clean`. En pratique : écris `frame` sans putText (l'overlay n'est affiché qu'à l'écran). NB : `cv2.VideoWriter` utilse BGR, ok ici.
8. `cap.release()` / `writer.release()` / `cv2.destroyAllWindows()` dans un `finally`.
9. Gestion d'erreurs : caméra introuvable → message clair en français + suggestion `--image` ; **aucune** exception non gérée.

### 4.8 `main.py`
```python
def parse_args(argv=None) -> argparse.Namespace  # construit le Config + renvoie
def main() -> int
if __name__ == "__main__": sys.exit(main())
```
Arguments CLI (argparse, français pour -h/--help) :
- `--source` `webcam|image|video` (défaut `webcam`).
- `--image PATH` (filled par `--source image`), `--video PATH` (`--source video`), `--cam INT` (indice webcam).
- `--effect NAME` (défaut `identity`) — valable aussi en mode démo initial.
- `--effect-param KEY=VALUE` (répétable) : ex `--effect-param strength=0.5`.
- `--device auto|cuda|cpu`.
- `--demo SECONDS` (float) : cycle auto d'effets (NULL=off).
- `--clean` : pas d'overlay texte/FPS à l'écran.
- `--record PATH.mp4` : enregistrement MP4 (voir 4.7). Si `--record` et `--clean` non précisé : activer `clean` automatiquement (la vidéo doit rester propre).
- `--preview` : n'applique pas l'effet (debug).
- `--crop-size INT` (défaut 256).
- `--no-smooth` : désactive le lissage temporel (alpha=1).
- `--conf FLOAT` (défaut 0.6).
- `--fps N`.

Mode `image` : une seule itération sur `frame`, applique effet + fusion, écrit `out/<nom>.png` si aucune sortie, ou `--record out.png/jpeg` ; affiche aussi (si pas `--clean`) et attend une touche.
Mode `video` : boucle comme webcam mais source = fichier vidéo.

---

## 5. Contrat d'interface graphique

- Fenêtre : `cv2.WINDOW_NORMAL` pour redimensionnement libre. `cv2.namedWindow("SnapFace", cv2.WINDOW_NORMAL)`.
- Overlay FPS : `cv2.putText(frame, f"SnapFace - {effect_name} - {fps:.0f} FPS", (12, 32), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)`.
- Camera solo sur poste sans écran (headless) : `--clean` calme les warnings X11, mais `cv2.imshow` nécessite un display — c'est OK sur le poste de l'utilisateur (bureau). Documenter ce point dans README.

---

## 6. Script de vérification automatique (`scripts/generate_samples.py`)

**Objectif : prouver que TOUT fonctionne sans webcam.** Ce script est OBLIGATOIRE. Implémentation :

1. `--image PATH` en entrée (défaut : télécharge une photo de face libre de droits, ex. une des images de test OpenCV `cv2` `samples/lena.jpg` n'existe plus → télécharger ex. `https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg`, sinon `samples/data/messi5.jpg`), sinon **génère** une image de test synthétique (visage schématique ou bruit) si aucune source.
2. Pour chaque effet de `available_effects()` avec ses params par défaut : applique le pipeline complet (detect → crop → effect → paste) et sauvegarde `samples/<effet>_<timestamp>.png`.
3. Ecrit `samples/index.html` listant les images générées (pour inspection visuelle rapide).
4. Imprime un résumé : `[OK] identity   -> samples/identity_....png` etc.
5. Séparément, teste `XAVIER` : si `--device cuda` indiqué, l'appel `torch.cuda.is_available()` doit être vrai, sinon warning (mais pas d'échec : CPU OK).

**Test de non-régression intégré** (assertions) : identité/taille/domaine :
- Pour chaque effet : le résultat de `from_tensor(apply(to_tensor(crop)))` a la même forme que `crop`.
- `np.max(np.abs(identity_result - crop)) < 1e-3`.
- `np.min >= 0` et `np.max <= 1` sur le tensor avant `from_tensor`.

---

## 7. Exécution & vérifications (commandes exactes)

```bash
cd /home/jose/projects/snap_face
PYTHON=./.venv/bin/python

$PYTHON -c "import torch, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
# attendu:  2.14.0 True 5.0.0   (si cuda False, NE PAS corriger, noter dans VERIFICATION.md)

$PYTHON scripts/generate_samples.py --dir samples
# attendu: [OK] pour chaque effet + index.html

# test image (headless d'image) :
$PYTHON main.py --source image --image /chemin/vers/une_photo.jpg --effect bulge --clean --record out/img_ince.png
# attendu: fichier out/img_ince.png créé, non vide.

# test vidéo courte sans webcam (mode démo + record) :
$PYTHON main.py --source image --image /chemin/vers/une_photo.jpg --demo 3 --clean --record out/demo.mp4
# (en mode image le demo s'applique à l'identique — mais le script prouve le chemin d'enregistrement)
```

---

## 8. Règles d'écriture de code

- **Commentaires** : minimum. Uniquement des docstrings courts (1 ligne) sur classes et méthodes utiles. Pas de commentaires vaseux.
- **Types** : annotations de types sur toutes les signatures (incl. numpy/torch).
- **Imports** : `import cv2`, `import numpy as np`, `import torch`, `import torch.nn.functional as F`.
- **Aucune dépendance supplémentaire** (pas d'onnxruntime : `cv2.FaceDetectorYN` lit l'ONNX directement).
- **Erreurs** : toujours propager avec messages en français clairs (`raise ValueError(f"Effet inconnu: {name}")`...).
- **Découpage** : garder chaque module < ~220 lignes.
- **Trailing whitespace** absent. Ruff non utilisé (pas installé), reste soigné.
- Ne pas ajouter d'emojis, ne pas créer de fichiers `.md` non demandés (sauf `VERIFICATION.md` si échec).

---

## 9. Workflow git

- L'init du dépôt est déjà fait (branche `main`, remote `origin` = `git@github.com:THEGENTLEMAN31/snapface.git`).
- **Commits** : plusieurs commits propres par phase (pas de GPG configuré sur ce poste, ne pas tenter de signature). Messages en anglais impératif court : `feat(detector): add YuNet face detection`, `feat(effects): bulge and pinch radial warps`, etc. **Jamais `--amend` ni force-push.**
- **Push final** : quand toutes les phases sont vertes, `git push -u origin main`.
- Si le push échoue (authentification SSH non configurée sur CE poste) : ne pas chercher à créer une clé, écrire dans `VERIFICATION.md` que le push est à faire à la main, et garder l'état local commité.

---

## 10. ORDRE D'EXÉCUTION (phases obligatoires)

1. **Phase 0 – Sanité** : vérifier les imports + CUDA (commande section 7). Écrire `notebook` de vérification si besoin. Si `opencv-python` manquant, l'installer (`uv pip install --python .venv/bin/python opencv-python numpy`) et relancer.
2. **Phase 1 – Structure** : créer tous les modules de la section 3 (vides), `snapface/__init__.py`, écrire `config.py`.
3. **Phase 2 – Effets** (les plus autonomes) : `base.py`, `identity.py`, `bulge.py`, `pinch.py`, `swirl.py`, `mirror.py`, `kaleidoscope.py`, `__init__.py` registre. Vérifier avec un petit test python inline que chaque `apply()` marche sur un tensor factice (forme, bornes).
4. **Phase 3 – Détection** : `detector.py` + `geometry.py`. Vérifier `detector.detect(frame)` sur l'image de test.
5. **Phase 4 – Pipeline + CLI** : `pipeline.py`, `main.py`.
6. **Phase 5 – Script de test** : `scripts/generate_samples.py`.
7. **Phase 6 – Vérification finale** : exécuter tout le bloc « commandes exactes » section 7 ; compiler avec `python -m compileall .` ; `git add . ; git commit ...` par lot ; push final.

## 11. Livrables finaux attendus

- Code fonctionnel conforme au présent spec.
- `VERIFICATION.md` absent si tout est vert (sinon présent avec la liste des échecs + sorties).
- Dépôt git avec historique propre et push effectué (ou note dans `VERIFICATION.md` si l'auth SSH bloque).
- README.md final (FR) : description courte, installation (`.venv`), utilisation (touches 1-6, q, mode démo, enregistrement), date du jour, « utilisé avec RTX 3050 / CUDA ».