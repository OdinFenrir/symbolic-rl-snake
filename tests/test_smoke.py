import unittest

from snake.cli import run


class SmokeTest(unittest.TestCase):
    def test_headless_runs(self):
        rc = run(
            num_games=1,
            render=False,
            load_state=False,
            debug=False,
            seed=42,
            max_steps=200,
            log_jsonl=None,
            save_every=None,
            state_dir=None,
            no_save=True,
        )
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
