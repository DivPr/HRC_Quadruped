#!/usr/bin/env python3
"""Verify your Stage 0 setup and tell you exactly what to fix.

    uv run python scripts/check_setup.py

Exits 0 if everything the first four stages need is working, 1 otherwise. Every
failure prints a concrete fix, not just a complaint.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED = {
    "mujoco": "3.13.0",
    "mujoco.mjx": "3.13.0",
    "jax": "0.8.2",
    "brax": "0.14.2",
    "mujoco_playground": None,
    "flax": "0.12.2",
    "numpy": "2.4.6",
    "ml_collections": "1.1.0",
}
PASS, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_problems: list[str] = []


def report(status: str, label: str, detail: str = "", fix: str = "") -> None:
    """Print one aligned status line, recording failures for the exit code."""
    print(f"[{status}] {label:<34} {detail}")
    if status == FAIL:
        _problems.append(f"{label}: {fix or detail}")
        if fix:
            print(f"         -> {fix}")


def check_python() -> None:
    """Require CPython 3.11 or 3.12; 3.13 has no jaxlib wheel for these pins."""
    version = platform.python_version()
    ok = sys.version_info[:2] in {(3, 11), (3, 12)}
    report(PASS if ok else FAIL, "python", f"{version} ({platform.machine()})",
           "install Python 3.11: `uv python install 3.11 && uv sync`")


def check_packages() -> None:
    """Import every pinned package and compare its version to pyproject.toml."""
    for name, expected in REQUIRED.items():
        try:
            module = importlib.import_module(name)
            if "." in name:
                module = importlib.import_module(name.split(".")[0])
        except ImportError as error:
            report(FAIL, name, str(error), "run `uv sync` from the repository root")
            continue
        found = getattr(module, "__version__", "?")
        if expected is None or found in ("?", expected):
            report(PASS, name, found)
        else:
            report(WARN, name, f"{found} (pinned {expected})")


def check_jax_backend() -> None:
    """Report the JAX backend; CPU is fine for Stages 1-3 and 5."""
    try:
        import jax
    except ImportError:
        return
    backend = jax.default_backend()
    devices = ", ".join(str(device) for device in jax.devices())
    if backend == "gpu":
        report(PASS, "jax backend", f"gpu -- {devices}")
    else:
        report(PASS, "jax backend", f"{backend} -- {devices}")
        print("         -> Stage 4 full training needs a GPU: use "
              "notebooks/train_pup_colab.ipynb")


def check_model() -> None:
    """Load both scenes and confirm the MJX one can be put on device."""
    try:
        import mujoco
        from mujoco import mjx

        from pup.envs.constants import MJX_SCENE, SCENE
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        report(PASS, "pup/assets/scene_flat.xml",
               f"nq={model.nq} nv={model.nv} nu={model.nu} "
               f"mass={model.body_mass.sum():.1f} kg")
        mjx_model = mjx.put_model(mujoco.MjModel.from_xml_path(str(MJX_SCENE)), impl="jax")
        report(PASS, "mjx.put_model", f"impl=jax, {mjx_model.nq} generalized coordinates")
    except Exception as error:  # noqa: BLE001 - the point is to report anything
        report(FAIL, "pup model", repr(error),
               "run from the repository root, and `uv pip install -e .`")


def check_render_backend() -> None:
    """Try one offscreen render, which Stage 4's GIFs and the docs media need."""
    try:
        import mujoco

        from pup.envs.constants import SCENE
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data)
            renderer.render()
        report(PASS, "offscreen rendering", f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'auto')}")
    except Exception as error:  # noqa: BLE001
        report(WARN, "offscreen rendering", repr(error))
        print("         -> headless Linux: `export MUJOCO_GL=egl` (or osmesa). "
              "Only Stage 4 rendering needs this.")


def check_docker() -> None:
    """Stage 5 needs Docker (or a native ROS 2 install); warn, do not fail."""
    binary = shutil.which("docker")
    if binary is None:
        report(WARN, "docker", "not found")
        print("         -> Stage 5 only: install Docker Desktop, or use native ROS 2 "
              "(see docs/05_ros2_sim2sim.md)")
        return
    try:
        version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                 timeout=15, check=True).stdout.strip()
        subprocess.run([binary, "info"], capture_output=True, timeout=30, check=True)
        report(PASS, "docker", version)
    except Exception:  # noqa: BLE001
        report(WARN, "docker", "installed but the daemon is not responding")
        print("         -> start Docker Desktop / `sudo systemctl start docker`")


def main() -> int:
    """Run every check and return a shell exit code."""
    print(f"Pup onboarding setup check -- {platform.platform()}")
    print(f"repository: {Path(__file__).resolve().parents[1]}\n")
    check_python()
    check_packages()
    check_jax_backend()
    check_model()
    check_render_backend()
    check_docker()
    print()
    if _problems:
        print(f"{len(_problems)} problem(s) to fix:")
        for problem in _problems:
            print(f"  - {problem}")
        return 1
    print("All good. Start with docs/01_mujoco_and_pd.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
