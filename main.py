"""Point d'entrée CLI SnapFace (compatibilité : `python main.py`)."""
import sys

from snapface.cli import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    sys.exit(main())
