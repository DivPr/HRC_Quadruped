"""Run a trained Brax policy with numpy alone -- no JAX, no GPU, no jit.

This is what the ROS 2 node imports. It must reproduce Brax's evaluation-time
inference exactly:

1. normalize:  ``x = (obs - obs_mean) / obs_std``
2. hidden layers: ``x = swish(x @ kernel_i + bias_i)`` where ``swish(x) = x *
   sigmoid(x)``
3. output layer (linear): ``logits = x @ kernel_last + bias_last``, shape (24,)
4. deterministic action: ``tanh(logits[:12])`` -- the second half is the
   Gaussian standard deviation and is unused at evaluation time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def swish(x: np.ndarray) -> np.ndarray:
    """Return x * sigmoid(x), the activation Brax's MLPs use by default."""
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    """A frozen Brax MLP policy evaluated in pure numpy."""

    def __init__(self, archive: dict) -> None:
        """Build from the dict of arrays produced by ``pup.train.export``."""
        # ===== TODO(student): Unpack the archive into layers and metadata =====
        raise NotImplementedError(
            "Stage 4: Unpack the archive into layers and metadata. See docs/04_training_with_brax.md")
        # ===== end TODO =====

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """Load a policy exported by ``pup.train.export.export_policy``."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({key: archive[key] for key in archive.files})

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to a (12,) action in [-1, 1], unitless."""
        # ===== TODO(student): Normalize, run the MLP, and squash with tanh =====
        raise NotImplementedError(
            "Stage 4: Normalize, run the MLP, and squash with tanh. See docs/04_training_with_brax.md")
        # ===== end TODO =====

    def joint_targets(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to (12,) joint position targets in rad."""
        # ===== TODO(student): Convert the action into absolute joint targets =====
        raise NotImplementedError(
            "Stage 4: Convert the action into absolute joint targets. See docs/04_training_with_brax.md")
        # ===== end TODO =====
