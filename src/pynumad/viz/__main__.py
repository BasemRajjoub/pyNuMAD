"""Entry point so ``python -m pynumad.viz`` works."""
import sys

from pynumad.viz.cli import main

if __name__ == "__main__":
    sys.exit(main())
