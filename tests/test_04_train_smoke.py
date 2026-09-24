import json
import time

import pytest

from pup.train.train_ppo import train


@pytest.mark.slow
def test_cpu_smoke(tmp_path):
    started = time.perf_counter()
    _, params, metrics = train(out=tmp_path, render=False)
    assert time.perf_counter()-started < 240
    assert (tmp_path / "policy.pkl").stat().st_size > 1000
    assert list((tmp_path / "checkpoints").iterdir())
    assert (tmp_path / "learning_curve.csv").stat().st_size > 100
    assert len(json.loads((tmp_path / "eval.json").read_text())["commands"]) == 5
    assert params[0].mean.shape == (45,)
    assert "eval/episode_reward" in metrics
