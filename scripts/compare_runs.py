from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def describe(path: Path, label: str) -> None:
    if not path.exists():
        print(f"warning: {path} not found")
        return
    df = pd.read_json(path, lines=True)
    cols = [
        "score",
        "forced_rate",
        "tuner_reward",
        "tuner_safety",
        "tuner_best",
        "rsm_prior_hits",
        "memory_size",
    ]
    print(f"\n--- {label} ({path.name}) ---")
    print(df[cols].describe())
    total_forced = df["safety_forced"].sum()
    total_steps = df["steps"].sum()
    if total_steps:
        rate = 100.0 * total_forced / total_steps
        print(f"session forced rate: {rate:.2f}% ({total_forced} forced / {total_steps} steps)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe JSONL benchmark runs")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("runs/train_preset.jsonl"),
        help="Training log to summarize",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("runs/eval_preset.jsonl"),
        help="Evaluation log to summarize",
    )
    args = parser.parse_args()

    describe(args.train, "training")
    describe(args.eval, "evaluation")


if __name__ == "__main__":
    main()
