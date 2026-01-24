from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid

from snake.cli import run


class DeterminismTest(unittest.TestCase):
    def _run_scores(self, seed: int, log_path: str, state_dir: str) -> list[int]:
        rc = run(
            num_games=3,
            render=False,
            load_state=False,
            debug=False,
            seed=seed,
            max_steps=200,
            log_jsonl=log_path,
            save_every=None,
            state_dir=state_dir,
            no_save=True,
        )
        self.assertEqual(rc, 0)
        with open(log_path, encoding="utf-8") as fh:
            return [json.loads(line)["score"] for line in fh]

    def test_headless_runs_are_deterministic(self):
        tmp_root = os.path.abspath(
            os.path.join(os.getcwd(), f"tmp-determinism-{uuid.uuid4().hex}")
        )
        os.makedirs(tmp_root, exist_ok=True)
        try:
            scores = []
            for run_id in range(2):
                log_path = os.path.join(tmp_root, f"run{run_id}.jsonl")
                state_dir = tmp_root
                scores.append(self._run_scores(2026, log_path, state_dir))
            self.assertEqual(scores[0], scores[1])
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
