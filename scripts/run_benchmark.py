import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_command(cmd: list[str]) -> None:
    print(f">> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def describe_jsonl(path: Path, label: str) -> None:
    if not path.exists():
        print(f"warning: {path} does not exist")
        return
    df = pd.read_json(path, lines=True)
    cols = ["score", "forced_rate", "tuner_safety", "tuner_reward", "memory_size"]
    print(f"\n--- {label} ({path.name}) ---")
    print(df[cols].describe())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train+eval benchmark for snake")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to fix during the benchmark",
    )
    parser.add_argument(
        "--train-games",
        type=int,
        default=300,
        help="Number of games in training run",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Number of games in evaluation run",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("state/benchmark"),
        help="State directory used for both runs",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Where to emit JSONL telemetry files",
    )

    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(exist_ok=True)

    train_log = args.runs_dir / f"train_seed{args.seed}.jsonl"
    eval_log = args.runs_dir / f"eval_seed{args.seed}.jsonl"

    cmd_train = [
        sys.executable,
        "-m",
        "snake",
        "--no-render",
        "--num-games",
        str(args.train_games),
        "--seed",
        str(args.seed),
        "--state-dir",
        str(args.state_dir),
        "--log-jsonl",
        str(train_log),
    ]

    cmd_eval = [
        sys.executable,
        "-m",
        "snake",
        "--no-render",
        "--num-games",
        str(args.eval_games),
        "--seed",
        str(args.seed),
        "--state-dir",
        str(args.state_dir),
        "--log-jsonl",
        str(eval_log),
        "--eval",
    ]

    run_command(cmd_train)
    run_command(cmd_eval)

    describe_jsonl(train_log, "training")
    describe_jsonl(eval_log, "evaluation")


if __name__ == "__main__":
    main()
