"""Revue automatique d'un rendu (vidéo ou image) par un modèle vision."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snapface.review import review_image, review_video, summarize


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Contrôle qualité d'un rendu SnapFace par un modèle vision.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Vidéo à analyser (Gemini, entrée native)")
    src.add_argument("--image", help="Image à analyser")
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", default=None, help="Écrit le verdict JSON à ce chemin")
    ap.add_argument("--strict", action="store_true",
                    help="Code de sortie 1 si le verdict est 'a_corriger'")
    args = ap.parse_args()

    try:
        if args.video:
            verdict = review_video(Path(args.video))
        else:
            verdict = review_image(Path(args.image), args.backend, args.model)
    except ValueError as e:
        print(f"Revue impossible: {e}")
        return 2

    print(summarize(verdict))
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(verdict, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return 1 if args.strict and verdict.get("verdict") != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
