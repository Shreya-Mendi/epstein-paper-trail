"""
main.py — Top-level entrypoint for Paper Trail.

Provides a simple CLI to run individual pipeline stages.
"""

import argparse
import subprocess
import sys


def run(script: str, args: list[str] = None) -> None:
    """Run a pipeline script as a subprocess.

    Args:
        script: Path to the Python script to run.
        args: Optional list of additional arguments.
    """
    cmd = [sys.executable, script] + (args or [])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Script {script} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="paper-trail",
        description="Paper Trail — Epstein Case NLP Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Run data collection (scripts/make_dataset.py)")
    subparsers.add_parser("features", help="Build features (scripts/build_features.py)")
    subparsers.add_parser("train", help="Train all models (scripts/model.py)")
    subparsers.add_parser("evaluate", help="Run evaluation (scripts/evaluate.py)")
    subparsers.add_parser(
        "serve",
        help="Start the FastAPI backend (uvicorn app.backend.main:app --reload)",
    )

    args = parser.parse_args()

    if args.command == "collect":
        run("scripts/make_dataset.py")
    elif args.command == "features":
        run("scripts/build_features.py")
    elif args.command == "train":
        run("scripts/model.py")
    elif args.command == "evaluate":
        run("scripts/evaluate.py")
    elif args.command == "serve":
        cmd = [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--reload"]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
