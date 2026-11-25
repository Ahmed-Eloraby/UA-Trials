"""Launch the command line. This script adds ``src`` to the import path, then runs ``main``.

Command-line arguments are those of :func:`makespan.__main__.main`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from makespan.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
