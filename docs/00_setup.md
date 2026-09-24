# Stage 0 — Setup

Important: **this repository uses two different Python environments.**

| | Stages 1–4 | Stage 5 |
|---|---|---|
| Interpreter | the `uv` virtual environment in `.venv/` | the ROS 2 container's **system** Python |
| Managed by | `uv sync` (`pyproject.toml` + `uv.lock`) | `apt` inside `docker/ros2/Dockerfile` |
| Has JAX/MJX/Brax? | yes | no, and it does not need it |
| How you run things | `uv run python ...` | `ros2 run ...` inside the container |

They never mix. ROS 2 is built against the system interpreter and breaks in
interesting ways if you try to `source` it inside a virtualenv. That is exactly
why Stage 5's policy runs on pure numpy: `pup/policy/` imports nothing that the
container does not already have.

## Your task

### 1. Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`uv` is a Rust reimplementation of pip + virtualenv + pyenv. It reads
`pyproject.toml`, resolves against the committed `uv.lock`, and is incredibly useful in managing dependencies. If you are not using it already, it is definitely worth learning now.

### 2. Create the environment

Before cloning, make sure you have already created your OWN version of the template repository.

```bash
git clone https://github.com/<your-username>/onboarding-fall26.git
cd onboarding-fall26
uv python install 3.11
uv sync
```

### 3. Check it

```bash
uv run python scripts/check_setup.py
```

You want every line to say `ok`. The script checks your Python version, each
pinned package, which JAX backend you have (`cpu` is fine), that `pup/assets/scene_flat.xml` loads and `mjx.put_model` succeeds, that
offscreen rendering works, and that Docker is alive. Anything that fails prints
the fix on the next line.

### 4. Run the tests once

```bash
uv run pytest -m "not slow and not gpu and not ros"
```

Almost everything will report **▷ NOT STARTED**. The student
functions all currently raise `NotImplementedError` and it is your job to implement them.

```bash
uv run python scripts/progress.py
```

### 5. (Stage 5 only) Get Docker working

```bash
docker compose -f docker/compose.yaml build
```

## When you're finished

```bash
uv run python scripts/check_setup.py   # exits 0
```

## Platform notes

**Linux (native, x86_64 + NVIDIA)** — the best case. `uv sync --extra gpu`
gives you a local CUDA JAX and you can run Stage 4 without Colab. The MuJoCo
viewer, Docker, and ROS 2 all work natively.

**macOS (Apple Silicon)** — everything in Stages 1–3 and 5 works. JAX is
CPU-only, which is fine: the MJX tests are sized for CPU. Stage 4's full
training goes to Colab. **The passive MuJoCo viewer needs `mjpython`, not
`python`**, because of how macOS handles GUI event loops:

```bash
uv run mjpython -c "from pup.sim.standup import stand_up; stand_up(headless=False)"
```

Headless runs (all the tests) use plain `python`.

**Windows + WSL2** — install everything inside WSL, not in Windows Python. The
GUI viewer works through WSLg on Windows 11; on Windows 10 run headless and
render GIFs instead. Docker Desktop with the WSL2 backend handles Stage 5. Keep
the repository on the Linux filesystem (`~/onboarding-fall26`), not under
`/mnt/c/`.

**Headless Linux servers** — set `MUJOCO_GL=egl` if you have a GPU, or
`MUJOCO_GL=osmesa` for software rendering, before anything imports MuJoCo.

## GPU / Colab

- Local NVIDIA: `uv sync --extra gpu`, then confirm with
  `uv run python -c "import jax; print(jax.default_backend())"` → `gpu`.
- Everyone else: Stage 4 uses `notebooks/train_pup_colab.ipynb` on a free T4.
  Nothing else in the repository needs a GPU.

## Common mistakes

- **Running `python` instead of `uv run python`.** You will get your system
  interpreter, which has none of this installed.
- **Running from the wrong directory.** All commands assume the repository root.
- **`uv sync` inside the ROS 2 container.** Don't. The container has its own
  Python on purpose.
- **Python 3.13.** `jaxlib==0.8.2` has no wheel for it. Use 3.11.