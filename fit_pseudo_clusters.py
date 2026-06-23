"""Run fit pseudo clusters."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cli_fit_pseudo_clusters import main  # noqa: E402


if __name__ == "__main__":
    main()
