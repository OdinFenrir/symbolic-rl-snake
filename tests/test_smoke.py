from snake.cli import run

def test_smoke_headless_runs():
    assert run(
        num_games=2,
        render=False,
        load_state=False,
        debug=False,
        seed=123,
        max_steps=200,
        log_jsonl=None,
        save_every=None,
        state_dir=None,
        no_save=True,
    ) == 0
