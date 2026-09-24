"""Train Pup with the same Brax wrapper pattern used by Playground."""

import argparse
import csv
import functools
import json
import time
from pathlib import Path

import jax
from brax.io import model as model_io
from brax.training.agents.ppo import networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from pup.envs.config import default_config
from pup.envs.pup_joystick import PupJoystick
from pup.envs.randomize import domain_randomize
from pup.train import ppo_params
from pup.train.evaluate import evaluate
from pup.train.render import render_rollout


def train(config_name: str = "cpu_smoke", out: str | Path = "runs/smoke",
          impl: str = "jax", seed: int = 0, render: bool = True,
          restore: str | Path | None = None) -> tuple:
    """Run PPO and save checkpoint, CSV, metrics JSON and an optional rollout GIF.

    Args:
      config_name: a function name in `ppo_params` (`full`, `cpu_smoke`, ...).
      out: run directory; checkpoints, CSV, JSON and GIF are written here.
      impl: MJX backend, "jax" or "warp".
      seed: PRNG seed for the whole run.
      render: write `rollout.gif` at the end (needs a working GL backend).
      restore: an Orbax checkpoint directory to resume from, e.g.
        `runs/colab/checkpoints/000030000000`. Useful when a Colab session dies.

    Returns:
      Inference factory, params tuple (normalizer, policy, value), and metrics.
      Environment observations (45,) and actions (12,) follow Stage 3 exactly.
    """
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = getattr(ppo_params, config_name)()
    environment_config = default_config()
    environment_config.impl = impl
    environment_config.episode_length = parameters["episode_length"]
    rows = []
    started = time.perf_counter()

    def progress(steps, metrics):
        row = {"steps": int(steps), "wall_seconds": time.perf_counter()-started,
               **{key: float(value) for key, value in metrics.items()}}
        rows.append(row)
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (output / "learning_curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)

    # ===== TODO(student): Wire the environment, wrappers, randomization and PPO networks =====
    raise NotImplementedError(
        "Stage 4: Wire the environment, wrappers, randomization and PPO networks. See docs/04_training_with_brax.md")
    # ===== end TODO =====
    model_io.save_params(str(output / "policy.pkl"), params)
    evaluation_config = default_config()
    evaluation_config.noise_config.level = 0.0
    evaluation_config.episode_length = parameters["episode_length"]
    evaluation_config.impl = impl
    evaluation_env = PupJoystick(evaluation_config)
    report = evaluate(evaluation_env, inference_fn, params,
                       n_episodes=1 if config_name == "cpu_smoke" else 5, seed=seed)
    (output / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(dict(config=config_name, seed=seed, impl=impl,
        wall_seconds=time.perf_counter()-started, devices=[str(d) for d in jax.devices()],
        trained_walking_policy=report["walking_passes"]), indent=2) + "\n")
    if render:
        render_rollout(evaluation_env, inference_fn, params, output / "rollout.gif", seed=seed)
    return inference_fn, params, metrics


def main() -> None:
    """Parse command-line configuration and start a reproducible training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["PupJoystickFlat"], default="PupJoystickFlat")
    parser.add_argument("--config", choices=["full", "cpu_smoke", "t4_fast", "cpu_reference"], default="cpu_smoke")
    parser.add_argument("--impl", choices=["jax", "warp"], default="jax")
    parser.add_argument("--out", default="runs/smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--restore", default=None,
                        help="Orbax checkpoint directory to resume from")
    args = parser.parse_args()
    train(args.config, args.out, args.impl, args.seed, not args.no_render,
          restore=args.restore)


if __name__ == "__main__":
    main()
