"""Interface en ligne de commande SnapFace."""
import argparse
import sys
from pathlib import Path

from .config import Config
from .pipeline import Pipeline


def _parse_effect_param(s: str) -> tuple[str, float]:
    key, _, val = s.partition("=")
    if not key or not val:
        raise argparse.ArgumentTypeError(f"Format invalide '{s}', attendu KEY=VALUE")
    try:
        return key, float(val)
    except ValueError:
        return key, val  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="snapface",
        description="Filtres faciaux temps réel (OpenCV + YuNet + effet PyTorch).")
    p.add_argument("--source", choices=["webcam", "image", "video"], default="webcam",
                   help="Source d'entrée (défaut: webcam)")
    p.add_argument("--image", type=str, help="Chemin image (avec --source image)")
    p.add_argument("--video", type=str, help="Chemin vidéo (avec --source video)")
    p.add_argument("--cam", type=int, default=0, help="Indice de la webcam (défaut: 0)")
    p.add_argument("--effect", type=str, default="identity",
                   help="Effet initial (identity, bulge, pinch, swirl, mirror, "
                        "kaleidoscope, big_eyes, big_mouth)")
    p.add_argument("--effect-param", action="append", default=[], type=_parse_effect_param,
                   metavar="KEY=VALUE", help="Paramètre d'effet, répétable (ex: strength=0.5)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Périphérique PyTorch (défaut: auto)")
    p.add_argument("--demo", type=float, default=None,
                   help="Mode démo : cycle auto d'effets toutes les N secondes")
    p.add_argument("--clean", action="store_true", help="Pas d'overlay texte/FPS à l'écran")
    p.add_argument("--record", type=str, help="Chemin de sortie vidéo (MP4)")
    p.add_argument("--preview", action="store_true", help="N'applique pas l'effet (debug)")
    p.add_argument("--crop-size", type=int, default=256, help="Résolution du crop (défaut: 256)")
    p.add_argument("--no-smooth", action="store_true", help="Désactive le lissage temporel")
    p.add_argument("--min-cutoff", type=float, default=1.2,
                   help="One Euro : coupure min en Hz (défaut: 1.2)")
    p.add_argument("--beta", type=float, default=0.06, help="One Euro : réactivité (défaut: 0.06)")
    p.add_argument("--conf", type=float, default=0.6, help="Seuil de confiance détection")
    p.add_argument("--fps", type=int, default=30, help="FPS cible pour l'enregistrement")
    p.add_argument("--seconds", type=float, default=None,
                   help="Durée d'enregistrement en secondes (puis arrêt propre)")
    p.add_argument("--det-every", type=int, default=2,
                   help="Détecter 1 frame sur N (suivi optique entre-temps)")
    p.add_argument("--debug-grid", action="store_true",
                   help="Affiche le maillage de déformation dans la vidéo")
    p.add_argument("--split", action="store_true", help="Split avant/après à l'écran")
    p.add_argument("--pulse", type=float, default=0.0,
                   help="Pulsation de l'effet, amplitude 0..1 (ex: 0.3)")
    p.add_argument("--morph", type=float, default=0.45,
                   help="Durée de transition entre effets, en secondes")
    p.add_argument("--no-h264", action="store_true",
                   help="Ne pas ré-encoder en H.264 à la fermeture")
    p.add_argument("--review", action="store_true",
                   help="Analyse la vidéo produite avec un modèle vision (Gemini)")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
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
        det_every=max(1, args.det_every),
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        debug_grid=args.debug_grid,
        split=args.split,
        pulse=args.pulse,
        morph_seconds=args.morph,
        seconds=args.seconds,
        export_h264=not args.no_h264,
        review=args.review,
    )


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.source == "image" and not args.image:
        print("--source image requiert --image PATH")
        return 1
    if args.source == "video" and not args.video:
        print("--source video requiert --video PATH")
        return 1
    cfg = config_from_args(args)
    try:
        code = Pipeline(cfg, source=args.source, image_path=args.image,
                        video_path=args.video).run()
    except ValueError as e:
        print(f"Erreur: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nArrêt par interruption (Ctrl-C).")
        return 0
    if code == 0 and cfg.review and cfg.record:
        from .review import review_video, summarize
        try:
            print(summarize(review_video(cfg.record)))
        except ValueError as e:
            print(f"Revue impossible: {e}")
    return code


if __name__ == "__main__":
    sys.exit(main())
