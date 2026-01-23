import unittest

from snake.cli import run


class SmokeTest(unittest.TestCase):
    def test_headless_runs(self):
        # Run a single short headless game to ensure nothing crashes.
        rc = run(num_games=1, render=False, load_state=False, debug=False, seed=123, max_steps=200)
        self.assertIn(rc, (0, 130))


if __name__ == "__main__":
    unittest.main()
