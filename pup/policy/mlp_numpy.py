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
        self.obs_mean = archive["obs_mean"]
        self.obs_std = archive["obs_std"]

        self.n_layers = int(archive["n_layers"])
        self.obs_size = int(archive["obs_size"])
        self.action_size = int(archive["action_size"])

        self.action_scale = archive["action_scale"]
        self.default_pose = archive["default_pose"]

        self.layers = [
            (archive[f"kernel_{i}"], archive[f"bias_{i}"])
            for i in range(self.n_layers)
        ]
        # ===== end TODO =====

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """Load a policy exported by ``pup.train.export.export_policy``."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({key: archive[key] for key in archive.files})

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to a (12,) action in [-1, 1], unitless."""
        # ===== TODO(student): Normalize, run the MLP, and squash with tanh =====
        x = (np.asarray(obs) - self.obs_mean) / self.obs_std

        for kernel, bias in self.layers[:-1]:
            x = swish(x @ kernel + bias)

        kernel, bias = self.layers[-1]
        logits = x @ kernel + bias

        return np.tanh(logits[:self.action_size])
        # ===== end TODO =====

    def joint_targets(self, obs: np.ndarray) -> np.ndarray:
        """Map a (45,) observation to (12,) joint position targets in rad."""
        # ===== TODO(student): Convert the action into absolute joint targets =====
        action = self(obs)
        return self.default_pose + self.action_scale * action
        # ===== end TODO =====
