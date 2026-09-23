"""Point d'entrée CLI SnapFace."""
import argparse
import sys
from pathlib import Path

from snapface.config import Config
from snapface.pipeline import Pipeline


def _parse_effect_param(s: str) -> tuple[str, float]:
    key, _, val = s.partition("=")
    if not key or not val:
        raise argparse.ArgumentTypeError(f"Format invalide '{s}', attendu KEY=VALUE")
    return key, float(val)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="snapface",
        description="Filtres faciaux temps réel (OpenCV + YuNet + effets PyTorch).")
    p.add_argument("--source", choices=["webcam", "image", "video"], default="webcam",
                   help="Source d'entrée (défaut: webcam)")
    p.add_argument("--image", type=str, help="Chemin image (avec --source image)")
    p.add_argument("--video", type=str, help="Chemin vidéo (avec --source video)")
    p.add_argument("--cam", type=int, default=0, help="Indice de la webcam (défaut: 0)")
    p.add_argument("--effect", type=str, default="identity",
                   help="Effet initial (identity, bulge, pinch, swirl, mirror, kaleidoscope)")
    p.add_argument("--effect-param", action="append", default=[], type=_parse_effect_param,
                   metavar="KEY=VALUE", help="Paramètre d'effet, répétable (ex: strength=0.5)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Périphérique PyTorch (défaut: auto)")
    p.add_argument("--demo", type=float, default=None,
                   help="Mode démo : cycle auto d'effets toutes les N secondes")
    p.add_argument("--clean", action="store_true", help="Pas d'overlay texte/FPS à l'écran")
    p.add_argument("--record", type=str, help="Chemin de sortie MP4 (enregistrement)")
    p.add_argument("--preview", action="store_true", help="N'applique pas l'effet (debug)")
    p.add_argument("--crop-size", type=int, default=256, help="Résolution du crop (défaut: 256)")
    p.add_argument("--no-smooth", action="store_true", help="Désactive le lissage temporel de la bbox")
    p.add_argument("--conf", type=float, default=0.6, help="Seuil de confiance détection (défaut: 0.6)")
    p.add_argument("--fps", type=int, default=30, help="FPS cible pour l'enregistrement (défaut: 30)")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    cfg = Config(
        cam_index=args.cam,
        crop_size=args.crop_size,
        effect=args.effect,
        device=args.device,
        demo_seconds=args.demo,
        clean=args.clean or args.record is not None,
        record=Path(args.record) if args.record else None,
        smooth_alpha=1.0 if args.no_smooth else 0.45,
        conf_threshold=args.conf,
        effect_params=dict(args.effect_param),
        preview=args.preview,
        fps_target=args.fps,
    )
    source = args.source
    if source == "image" and not args.image:
        print("--source image requiert --image PATH")
        return 1
    if source == "video" and not args.video:
        print("--source video requiert --video PATH")
        return 1
    try:
        return Pipeline(cfg, source=source, image_path=args.image, video_path=args.video).run()
    except ValueError as e:
        print(f"Erreur: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nArrêt par interruption (Ctrl-C).")
        return 0


if __name__ == "__main__":
    sys.exit(main())