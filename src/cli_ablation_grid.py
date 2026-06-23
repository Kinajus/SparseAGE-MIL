"""Generate ablation configs for K sensitivity and topology-prior analyses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from pathlib import Path

import yaml

from sparseage_ablation import generate_ablation_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or run ablation configs."
    )
    parser.add_argument("--template", required=True, help="Base YAML config.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run", action="store_true", help="Run python train.py for each generated config."
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Optional fold override when running."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.template).open("r", encoding="utf-8") as handle:
        template = yaml.safe_load(handle)
    paths = generate_ablation_configs(template, args.output_dir)
    print("Generated ablation configs:")
    for path in paths:
        print(path)
    if args.run:
        for path in paths:
            cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "train.py"), "--config", str(path)]
            if args.fold is not None:
                cmd.extend(["--fold", str(args.fold)])
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
