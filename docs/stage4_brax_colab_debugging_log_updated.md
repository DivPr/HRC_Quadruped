# Stage 4 Colab / Brax PPO Debugging Log

## Context

This document records the debugging process for **Stage 4 --- Training
with Brax PPO** in the Purdue Humanoid Robotics Club quadruped
onboarding project.

The target training command was:

``` bash
uv run python -m pup.train.train_ppo \
  --env PupJoystickFlat \
  --config t4_fast \
  --seed 0 \
  --out runs/colab
```

The intended `t4_fast` configuration was:

-   30,000,000 environment steps
-   2,048 parallel environments
-   5 evaluations
-   32 evaluation environments
-   otherwise inherited from the full Go1 PPO configuration
-   large policy/value MLPs inherited from the full configuration

The final result was successful:

``` text
walking_passes: True
```

The final evaluation had a **0% fall rate on every tested command** and
a mean episode length of **1000** for every command.

------------------------------------------------------------------------

## 1. Initial Problem

The main symptom was that `t4_fast` appeared to hang.

The Colab training cell would start:

``` python
subprocess.run(
    ["uv", "run", "python", "-m", "pup.train.train_ppo",
     "--env", "PupJoystickFlat",
     "--config", CONFIG,
     "--seed", str(SEED),
     "--out", "runs/colab"],
    check=True,
)
```

but there would be **no evaluation output for a long time**.

The concerns raised during debugging included:

-   "Why is there no output?"
-   "Is PPO actually training?"
-   "Why is the T4 at 100% utilization but nothing prints?"
-   "Why did I previously let it run for 20--30 minutes without this
    working?"
-   "Why is the process suddenly CPU-bound?"
-   "Is my Stage 4 implementation wrong even though it passes?"
-   "Should I just use the provided/reference `.npz` policy?"
-   "Why did this run work when the earlier ones did not?"

A major source of confusion was that the notebook says:

> Expect tens of minutes. Each line of output is one evaluation.

Because output was sparse, lack of terminal output looked like a hang
even when the process was still doing work.

------------------------------------------------------------------------

## 2. Stage 4 Implementation Verification

The final `train_ppo.py` wiring was checked and found to be clean.

The relevant implementation was:

``` python
environment = PupJoystick(environment_config)

network_factory = functools.partial(
    networks.make_ppo_networks,
    **parameters.pop("network_factory"),
)

inference_fn, params, metrics = ppo.train(
    environment=environment,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=domain_randomize,
    network_factory=network_factory,
    seed=seed,
    progress_fn=progress,
    save_checkpoint_path=str(output / "checkpoints"),
    restore_checkpoint_path=str(Path(restore).resolve()) if restore else None,
    **parameters,
)
```

This matched the Stage 4 requirements:

1.  Construct `PupJoystick`.
2.  Build the PPO network factory.
3.  Use Playground's `wrap_for_brax_training`.
4.  Pass `domain_randomize`.
5.  Pass the seed.
6.  Pass the progress callback.
7.  Configure checkpoint save/restore.
8.  Forward the PPO configuration parameters.

No final debug hacks remained in this file.

The command-line configuration was also clean and accepted only:

-   `full`
-   `cpu_smoke`
-   `t4_fast`
-   `cpu_reference`

### Conclusion

The Stage 4 implementation itself was **not the failure**.

------------------------------------------------------------------------

## 3. PPO Configuration Verification

The configuration file was checked.

### `full`

It used:

``` python
locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
```

with:

``` python
config["network_factory"]["value_obs_key"] = "state"
```

The inherited configuration included approximately:

``` text
num_timesteps = 200,000,000
num_envs = 8192
episode_length = 1000
unroll_length = 20
batch_size = 256
num_minibatches = 32
num_updates_per_batch = 4

policy_hidden_layer_sizes = (512, 256, 128)
value_hidden_layer_sizes  = (512, 256, 128)
```

### `t4_fast`

The actual fallback configuration was:

``` python
config.update(
    num_timesteps=30_000_000,
    num_envs=2048,
    num_evals=5,
    num_resets_per_eval=0,
    num_eval_envs=32,
)
```

Everything else was inherited from `full`.

### Important observation

`t4_fast` was not a tiny test configuration. It still used the large PPO
networks and most of the full training settings.

------------------------------------------------------------------------

## 4. CPU Smoke Test

Before investigating the T4 behavior, the CPU smoke test was run
successfully:

``` bash
uv run python -m pup.train.train_ppo \
  --config cpu_smoke \
  --out runs/smoke \
  --no-render
```

It completed successfully and produced:

-   checkpoints
-   `eval.json`
-   `learning_curve.csv`
-   `policy.pkl`
-   `run.json`

The smoke configuration intentionally used:

``` text
20,000 requested timesteps
8 environments
episode length 100
small (32, 32) policy/value networks
```

The smoke run did **not** need to produce a walking policy. Its purpose
was to prove that the training pipeline functioned.

Later, the smoke-test CSV was accidentally considered while discussing
the T4 run. Its `episode_length=100` immediately identified it as
smoke/debug output rather than real `t4_fast` output.

------------------------------------------------------------------------

## 5. First T4 Attempts

Multiple `t4_fast` attempts appeared to sit without output for 20--30
minutes.

During one of these attempts, `nvidia-smi` showed approximately:

``` text
Tesla T4
~3359 MiB GPU memory
GPU utilization: 100%
P0
~67 W / 70 W
```

This proved that at least during that observation, the GPU was actively
computing.

However, because no progress JSON had appeared, it was unclear whether:

-   training was progressing normally,
-   JAX was compiling,
-   the initial evaluation was taking a long time,
-   the process was stuck,
-   or output was simply too sparse.

These runs were stopped before completion.

------------------------------------------------------------------------

## 6. Environment Reset / Missing `.venv`

At one point, a Colab runtime no longer contained the expected virtual
environment.

A command failed with:

``` text
.venv/bin/python: No such file or directory
```

and the Colab system Python produced:

``` text
ModuleNotFoundError: No module named 'mujoco_playground'
```

This was not a PPO bug. The runtime had simply lost/recreated its
environment.

The solution was to rerun the notebook setup:

``` python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "uv"],
    check=True,
)

subprocess.run(
    ["uv", "sync", "--extra", "gpu", "--python", "3.11"],
    check=True,
)
```

A useful lesson from this was:

> Do not install Brax/MuJoCo Playground manually into Colab's system
> Python. Use the repository's `.venv`/`uv` environment.

------------------------------------------------------------------------

## 7. Environment and Reset Timing Diagnostics

The environment itself was tested independently of PPO.

### Base Pup environment

Observed timing:

``` text
construction: ~0.32 s
first JIT reset: ~26.95 s
observation shape: (45,)
backend: GPU
```

### 2048-environment randomized wrapper

Construction took approximately:

``` text
~1.68 s
```

### Wrapped reset

A 2048-environment wrapped reset took approximately:

``` text
~38.0 s
observation shape: (2048, 45)
```

### Conclusion

Environment construction and reset were not enough to explain a 20--30
minute apparent stall by themselves.

------------------------------------------------------------------------

## 8. Domain Randomization Diagnostic Mistake

An early wrapper diagnostic incorrectly passed raw `domain_randomize`
directly in a way that did not match how Brax binds the randomization
RNG.

The relevant signature was effectively:

``` python
domain_randomize(model, rng)
```

The Playground/Brax wrapper handles the RNG binding before applying the
randomization function.

The diagnostic was corrected, and the proper randomized wrapper/reset
worked.

This was a debugging mistake rather than a project bug.

------------------------------------------------------------------------

## 9. Inspecting Playground's Wrapper

The installed Playground wrapper was inspected.

Conceptually, it performed:

``` python
if randomization_fn is None:
    env = VmapWrapper(env)
else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)

env = EpisodeWrapper(env, episode_length, action_repeat)
env = BraxAutoResetWrapper(env, full_reset=full_reset)
```

This confirmed that Stage 4 was using the expected vectorization,
episode handling, reset behavior, and domain randomization structure.

------------------------------------------------------------------------

## 10. Investigating the First Evaluation

Brax's evaluator/training source was inspected.

One important discovery was that an evaluation can occur before the
first visible progress callback.

That suggested a possible reason for long initial silence:

``` text
start PPO
    ↓
wrap/vectorize environment
    ↓
JIT/compile
    ↓
initial evaluation
    ↓
first progress callback
```

Since the real episode length is 1000, an initial evaluation is
substantially more expensive than the 100-step smoke/debug evaluations.

However, this alone was never proven to explain all of the earlier
20--30 minute runs.

------------------------------------------------------------------------

## 11. Temporary `t4_debug` Configuration

A temporary configuration was created to determine whether GPU PPO could
complete at all.

It was approximately:

``` python
def t4_debug():
    config = t4_fast()
    config.update(
        num_timesteps=100_000,
        episode_length=100,
        num_evals=2,
    )
    return config
```

This run completed successfully in about five minutes.

Observed metrics included:

``` text
step 0:
eval/epoch_eval_time ≈ 61.85 s
eval/sps ≈ 51.73
wall_seconds ≈ 140.34

step 163840:
eval/epoch_eval_time ≈ 49.49 s
eval/sps ≈ 64.66
training/sps ≈ 2989
training/walltime ≈ 54.81 s
wall_seconds ≈ 245.28
```

The requested 100k timesteps became 163,840 reported steps because PPO
operates in whole rollout/update batches.

### Important mistake

The `training/sps ≈ 2989` value was briefly extrapolated to estimate a
multi-hour `t4_fast` run.

That extrapolation was not reliable because the short debug run included
startup/JIT effects and did not represent steady-state long-run
throughput.

That estimate was subsequently discarded.

------------------------------------------------------------------------

## 12. Temporary Debugging Changes That Caused More Problems

Several temporary debugging modifications created additional noise.

Temporary commits/configurations included names such as:

``` text
t4_debug
t4_benchmark
run_evals=False
```

At one point, CLI editing caused:

``` text
--config t4_benchmark
```

to be rejected because it was not an accepted argparse choice.

Another accidental edit broke the parser by leaving a standalone:

``` python
choices=[...]
```

instead of the intended:

``` python
parser.add_argument("--config", choices=[...])
```

This caused `--config` to become unrecognized.

Another edit inserted:

``` python
run_evals=False
```

without the necessary comma before:

``` python
**parameters
```

which produced an error resembling:

``` text
TypeError: unsupported operand type(s) for ** or pow(): 'bool' and 'dict'
```

These errors were caused by debugging edits, not by the original Stage 4
implementation.

The final `train_ppo.py` and `ppo_params.py` were restored to their
clean intended versions.

------------------------------------------------------------------------

## 13. Git / Stale Runtime Concerns

The Colab notebook only cloned the repository if:

``` python
Path("/content/pup-onboarding").exists()
```

was false.

That meant a runtime could retain an existing checkout across repeated
cell executions.

During debugging, repository state included temporary commits such as:

``` text
212216e choices
d18ee40 benchmark
b43bf99 run evals false
e399489 PPO for collab notebook
fd105b4 Stage 3 MJX Environment Finished
```

A rollback command discussed during debugging was:

``` bash
git reset --hard HEAD~3
git push --force-with-lease origin main
```

Eventually, the final working Stage 4 source was clean.

This made stale runtime/repository state one possible contributor to why
different Colab attempts behaved differently, although it was not proven
to be the cause.

------------------------------------------------------------------------

## 14. Inspecting the Official Playground Trainer

The installed `mujoco_playground.learning.train_jax_ppo.py` was
inspected.

The upstream trainer sets:

``` python
XLA_FLAGS += " --xla_gpu_triton_gemm_any=True"
XLA_PYTHON_CLIENT_PREALLOCATE = "false"
MUJOCO_GL = "egl"
```

It also measures the time before the first progress callback and reports
it as:

``` text
Time to JIT compile
```

This supported the idea that a substantial amount of host-side work may
occur before the first visible training progress.

### XLA environment check

The current training environment initially reported:

``` text
XLA_FLAGS = None
XLA_PYTHON_CLIENT_PREALLOCATE = None
MUJOCO_GL = egl
backend = gpu
devices = [CudaDevice(id=0)]
```

A temporary experiment added the upstream XLA settings.

However, after five minutes there was still no output.

Later inspection of the actual downloaded successful notebook showed
that the successful run **did not contain these XLA changes**.

Therefore:

> The XLA flag experiment did not explain or fix the successful run.

------------------------------------------------------------------------

## 15. `JAX_LOG_COMPILES` Experiment

Another attempt launched training with:

``` text
JAX_LOG_COMPILES=1
```

with the goal of seeing which JAX computation was compiling.

No useful output appeared.

This diagnostic did not identify the bottleneck and was abandoned.

It was later removed from the notebook.

------------------------------------------------------------------------

## 16. Colab Execution Timer Appeared Frozen

During the successful run, the visible Colab execution timer appeared to
stop around:

``` text
53 seconds
```

This initially suggested that the notebook frontend/runtime might have
stalled.

Instead of immediately killing the process, the Colab terminal was used
to inspect it.

This turned out to be one of the most useful diagnostics.

------------------------------------------------------------------------

## 17. `nvidia-smi`: Process Alive but GPU Idle

During the successful run, `nvidia-smi` showed:

``` text
Tesla T4
GPU memory: ~703 MiB
GPU utilization: 0%
power: ~29 W / 70 W
```

and a Python process was still holding the CUDA context:

``` text
/content/pup-onboarding/.venv/bin/python
```

This was very different from the earlier observation of 100% GPU
utilization.

At first this looked like evidence of a stalled training process.

------------------------------------------------------------------------

## 18. `ps`: The Process Was Actually CPU-Bound

The process was inspected with:

``` bash
ps -p 12898 -o pid,stat,etime,time,%cpu,%mem,cmd
```

The result was approximately:

``` text
PID    STAT   ELAPSED   TIME      %CPU
12898  Rl     11:42     00:11:19  96.8
```

This was extremely informative.

It showed:

-   the process was alive;
-   `R` meant it was actively running;
-   it had accumulated almost as much CPU time as wall-clock time;
-   it was using approximately one full CPU core;
-   it was not simply sleeping or deadlocked.

`top -H -p 12898` further showed that essentially one thread was
actively consuming CPU while many other threads were idle.

### Interpretation

At that moment, the workload was CPU-bound even though it was a GPU
training job.

Possible causes included JAX tracing/XLA compilation or other host-side
work.

The exact function was not captured, so it would be incorrect to claim
definitively that the entire period was XLA compilation.

------------------------------------------------------------------------

## 19. Why CPU Work Can Appear in a GPU JAX Job

A useful mental model is:

``` text
Python/JAX program
      ↓
trace Python computation             CPU
      ↓
construct/optimize computation       CPU
      ↓
XLA compilation                      CPU
      ↓
dispatch compiled executable
      ↓
GPU execution                        GPU
```

Therefore GPU utilization does not have to remain at 100% for the entire
lifetime of a JAX program.

The successful run itself demonstrated that a period of:

``` text
~97% CPU
0% GPU
```

did **not** mean the run was dead.

------------------------------------------------------------------------

## 20. The Run Suddenly Completed

After all of the above, the original training call returned:

``` text
CompletedProcess(
    args=[
        'uv', 'run', 'python', '-m', 'pup.train.train_ppo',
        '--env', 'PupJoystickFlat',
        '--config', 't4_fast',
        '--seed', '0',
        '--out', 'runs/colab'
    ],
    returncode=0
)
```

This proved that the supposedly stuck process had successfully
completed.

No special PPO algorithm change was responsible.

------------------------------------------------------------------------

## 21. Final Evaluation Result

The resulting `eval.json` reported:

``` text
walking_passes: True
```

### Command: `[0.5, 0.0, 0.0]`

``` text
mean_speed ≈ 0.5238
fall_rate = 0.0
mean_episode_length = 1000
```

### Command: `[1.0, 0.0, 0.0]`

``` text
mean_speed ≈ 1.0164
fall_rate = 0.0
mean_episode_length = 1000
```

### Command: `[0.0, 0.5, 0.0]`

``` text
mean_abs_vy ≈ 0.5592
mean_speed ≈ 0.5624
fall_rate = 0.0
mean_episode_length = 1000
```

### Command: `[0.0, 0.0, 1.0]`

``` text
fall_rate = 0.0
mean_episode_length = 1000
```

### Stand still: `[0.0, 0.0, 0.0]`

``` text
mean_speed ≈ 0.0642
fall_rate = 0.0
mean_episode_length = 1000
```

Every evaluation command completed without a fall.

The policy therefore satisfied the Stage 4 walking acceptance criterion.

------------------------------------------------------------------------

## 22. Reference `.npz` / Escape Hatch Discussion

Because training appeared unreliable for a while, using the provided
reference `.npz` policy was considered.

The intended plan would have been:

``` text
try real t4_fast
       ↓
if training cannot be completed
       ↓
use documented reference-policy escape hatch
       ↓
continue Stage 5
```

The important requirement would have been to document honestly that the
reference policy was used rather than claiming it was personally
trained.

Ultimately this was unnecessary because the user's own `t4_fast` policy
passed evaluation.

The user's own exported `pup_policy.npz` should therefore be used for
the remainder of the pipeline.

------------------------------------------------------------------------

## 23. Comparison With the Original Notebook

The original notebook was later provided verbatim.

Its training cell was simply:

``` python
subprocess.run(
    ["uv", "run", "python", "-m", "pup.train.train_ppo",
     "--env", "PupJoystickFlat",
     "--config", CONFIG,
     "--seed", str(SEED),
     "--out", "runs/colab"],
    check=True,
)
```

The downloaded working notebook was also inspected.

The successful run did **not** depend on:

``` text
XLA_FLAGS
XLA_PYTHON_CLIENT_PREALLOCATE
JAX_LOG_COMPILES
t4_debug
t4_benchmark
run_evals=False
```

Those were debugging experiments and were not the final fix.

### Critical conclusion

There was **no identified code-level fix** between the earlier
unsuccessful-looking runs and the final successful run.

The final clean Stage 4 implementation and intended `t4_fast`
configuration worked.

------------------------------------------------------------------------

## 24. Why Did the Final Run Work When Earlier 20--30 Minute Runs Did Not?

This remains the one unresolved question.

The earlier attempts were allowed to run for approximately 20--30
minutes without producing the same successful result, while the final
run completed.

What is known:

-   Stage 4 code was correct.
-   `t4_fast` configuration was correct.
-   CPU smoke passed.
-   short GPU PPO passed.
-   the T4 was visible to JAX.
-   the final clean run completed.
-   the final policy passed.
-   the successful run spent a long period CPU-bound.
-   the successful notebook did not contain a special XLA fix.

Possible explanations include:

### Different Colab runtime/host behavior

Colab sessions can differ in host CPU performance, load, runtime state,
compilation behavior, and other environmental details even when both
expose a Tesla T4.

This is plausible but was not directly measured.

### Different repository/runtime state

Earlier sessions contained temporary commits, debug configurations,
interrupted runs, and stale checkout/environment state.

The successful attempt occurred after returning to a clean
repository/environment.

This is also plausible but not proven to be the cause.

### Earlier runs may have been interrupted during a long silent phase

Because progress output was sparse, it was difficult to know whether an
earlier process was actually stuck.

At least some earlier runs may eventually have completed if left
running.

However, since some were allowed to run 20--30 minutes, it would be
incorrect to state with certainty that all of them would have succeeded.

### Final conclusion

The exact reason for the wall-clock difference between the earlier runs
and the successful run was **not captured with sufficient
instrumentation and cannot be determined retrospectively**.

------------------------------------------------------------------------

## 25. Debugging Mistakes / Things Not to Repeat

### Mistake 1: Treating no stdout as proof of a hang

The progress callback is sparse.

No output does not imply no work.

### Mistake 2: Restarting too aggressively

Repeatedly stopping runs destroyed potentially useful evidence and reset
expensive JAX startup work.

### Mistake 3: Changing multiple variables while debugging

Temporary configs, CLI edits, `run_evals`, XLA flags, and runtime resets
made it harder to identify causality.

### Mistake 4: Extrapolating performance from the tiny debug run

The short run's SPS included startup/compile behavior and was not a
reliable estimate for the full training job.

### Mistake 5: Using Colab's execution timer as authoritative process state

The timer appeared frozen even while the subprocess was actively
consuming CPU.

### Mistake 6: Assuming 0% GPU means the job is dead

The successful job itself spent a significant period with:

``` text
GPU utilization = 0%
CPU utilization ≈ 97%
```

and later completed successfully.

### Mistake 7: Running package inspection in Colab's system Python

The project's dependencies lived in:

``` text
/content/pup-onboarding/.venv/
```

not necessarily in the notebook kernel's Python environment.

### Mistake 8: Debugging the already-correct Stage 4 wiring

Once the smoke test and short GPU test passed, more evidence should have
been gathered from the running process before modifying the
implementation.

------------------------------------------------------------------------

## 26. Better Debugging Procedure for Next Time

If a future Brax/JAX training job appears stuck:

### 1. Do not modify the code immediately

First establish whether the process is alive.

``` bash
ps aux | grep '[p]up.train.train_ppo'
```

### 2. Check accelerator state

``` bash
nvidia-smi
```

Interpret it carefully:

-   high GPU utilization → active GPU work;
-   low GPU utilization + high CPU → potentially
    tracing/compilation/host work;
-   no process → job actually ended/died.

### 3. Inspect the process itself

``` bash
ps -p <PID> -o pid,stat,etime,time,%cpu,%mem,cmd
```

Useful states:

``` text
R = running
S = sleeping/waiting
D = uninterruptible wait
Z = zombie
```

### 4. Inspect threads if necessary

``` bash
top -H -p <PID>
```

### 5. Record timestamps before restarting

Capture:

``` text
wall time
CPU utilization
GPU utilization
GPU memory
process state
last progress step
repository commit
config name
```

### 6. Change one variable at a time

If an A/B experiment is necessary, keep everything else identical.

### 7. Preserve successful artifacts immediately

Once `walking_passes=true`, save:

``` text
results/learning_curve.png
results/eval_training.json
results/training.gif
results/pup_policy.npz
```

and download/archive the Colab run before the runtime disappears.

------------------------------------------------------------------------

## 27. Final Outcome

Despite appearing stuck multiple times, the clean `t4_fast` run
ultimately completed successfully.

Final state:

``` text
Stage 4 implementation: PASS
CPU smoke: PASS
GPU PPO execution: PASS
30M t4_fast training: PASS
Final evaluation: PASS
walking_passes: TRUE
Fall rate: 0% on all evaluation commands
Episode length: 1000 on all evaluation commands
Reference NPZ required: NO
```

The main unresolved issue is **why earlier Colab sessions behaved
substantially slower/differently than the successful session**.

No code-level fix was identified.

The most important lesson is that JAX/Brax GPU training can contain
long, poorly surfaced host-side phases, and sparse evaluation callbacks
make a healthy process look stalled. Future debugging should inspect the
running process before restarting or modifying a known-correct
implementation.

------------------------------------------------------------------------

# Follow-up --- Controlled `full` 200M Run and Silent Progress Logging

After the 30M `t4_fast` run succeeded with `walking_passes: true`, the
known-working notebook was saved separately. The active notebook was
then changed to `CONFIG = "full"` while keeping the same training
command, code, seed, environment, and general notebook workflow.

This was intentionally a much cleaner experiment than the earlier
debugging attempts. No `t4_debug`, `t4_benchmark`, `run_evals=False`,
`JAX_LOG_COMPILES`, or custom XLA flag was required.

## The successful runs did not visibly print evaluation steps

The notebook documentation says each evaluation should print a line.
That was not what was observed in the successful Colab runs.

The successful `t4_fast` run effectively looked like:

``` text
training cell starts
        ↓
no visible evaluation lines
        ↓
long-running subprocess
        ↓
CompletedProcess(..., returncode=0)
        ↓
eval.json → walking_passes=true
```

The `full` run behaved similarly. It remained visually silent during
training rather than continuously showing evaluation records.

This is a critical debugging observation:

> **Lack of visible notebook stdout was not evidence that PPO was not
> progressing.**

## `full` resource usage at approximately 10--11 minutes

A separate Colab terminal was used to inspect the running job without
interrupting it.

`nvidia-smi` showed:

``` text
Tesla T4
Temperature:       72 C
Performance state: P0
Power:             69 W / 70 W
GPU memory:        1221 MiB / 15360 MiB
GPU utilization:   100%
Python PID:         19313
```

`ps` showed that PID `19309` was only the lightweight `uv` launcher,
while PID `19313` was the actual `.venv` Python training process:

``` text
PID 19309  uv launcher                           ~0% CPU
PID 19313  actual Python training process         ~96.4% CPU
```

A focused snapshot showed:

``` text
PID:       19313
STAT:      Rl
ELAPSED:   11:05
CPU TIME:  00:10:42
CPU:       96.5%
MEM:       19.6%
```

Repeated `ps` calls showed CPU time advancing from approximately `10:22`
to `10:29` over about seven seconds.

Combined with `nvidia-smi`, the `full` process was therefore
simultaneously doing approximately:

``` text
CPU:       ~96–97%
GPU:       100%
GPU power: ~69 / 70 W
```

This demonstrated that high CPU usage did not mean JAX was using the CPU
*instead of* the GPU. The host CPU and T4 could both be heavily active
at the same time.

## Resource behavior across the different runs

  ---------------------------------------------------------------------------------------
  Run / observation              CPU          GPU   GPU memory        Power Observed
                                                                            result
  --------------------- ------------ ------------ ------------ ------------ -------------
  Earlier               not recorded         100%   \~3359 MiB    \~67/70 W No visible
  problematic-looking                                                       progress for
  `t4_fast`                                                                 20--30 min;
                                                                            killed

  Successful `t4_fast`       \~96.8%           0%    \~703 MiB    \~29/70 W Eventually
  at \~12 min                                                               completed and
                                                                            passed

  Successful `full` at     \~96--97%         100%   \~1221 MiB    \~69/70 W Actively
  \~10--11 min                                                              training;
                                                                            later passed

  Successful `full` at       \~98.4%         100%   \~1218 MiB    \~66/70 W Still
  \~39 min                                                                  actively
                                                                            progressing
  ---------------------------------------------------------------------------------------

This was one of the strangest observations of the entire debugging
process.

The successful `t4_fast` run was caught during a CPU-heavy/GPU-idle
phase. The successful `full` run was caught while both CPU and GPU were
heavily active. An earlier problematic-looking run was also caught at
100% GPU but was stopped before a successful result appeared.

Therefore neither CPU utilization nor GPU utilization alone was enough
to determine PPO progress.

## Using the logs to determine actual progress

At approximately 39 minutes into the `full` run, the process still
looked healthy:

``` text
Elapsed:      ~39:07
CPU:          ~98.4%
RAM:          ~19.7%
GPU util:     100%
GPU memory:   ~1218 MiB
GPU power:    ~66 / 70 W
GPU temp:     ~77 C
Process:      Rl
```

The notebook still was not printing evaluation lines.

Instead of restarting or modifying the run, progress was checked through
the files written by the trainer:

``` bash
ls -lah runs/colab
```

``` bash
find runs/colab/checkpoints -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | tail
```

and, most importantly:

``` bash
tail -10 runs/colab/learning_curve.csv
```

This revealed that the supposedly silent training run was actually
progressing normally.

The log showed:

``` text
160,563,200 / 200,000,000 steps
```

which is approximately:

``` text
80.3%
```

of the requested `full` training run.

### Progress recovered from `learning_curve.csv`

    Environment steps   Progress   Eval reward   Avg episode length
  ------------------- ---------- ------------- --------------------
                    0         0%          0.25                 99.7
               22.94M      11.5%         27.42                959.5
               45.88M      22.9%         29.66                962.8
               68.81M      34.4%         30.88                966.1
               91.75M      45.9%         31.24                974.1
              114.69M      57.3%         31.15                964.6
              137.63M      68.8%         30.71                956.8
              160.56M      80.3%         31.79                981.8

The transformation was substantial:

``` text
Step 0:
eval reward          ~0.25
avg episode length   ~99.7

160.6M:
eval reward          ~31.79
avg episode length   ~981.8 / 1000
```

The logged linear-velocity tracking metric also improved substantially,
with observed values approximately:

``` text
704 → 801 → 838 → 842 → 845 → 832 → 856
```

This established not merely that the process was alive, but that PPO was
advancing and the learned policy itself was improving.

## Monitoring hierarchy learned from the incident

Different diagnostics answered different questions:

``` text
ps
→ Is the Python subprocess alive and consuming CPU?

nvidia-smi
→ Is the GPU currently executing work?

learning_curve.csv / checkpoints
→ Is PPO actually advancing through environment steps?

learning-curve evaluation metrics
→ Is the policy improving?

eval.json
→ Did the final policy satisfy the acceptance criteria?
```

These questions should not be conflated.

A live process does not prove PPO is advancing. A busy GPU does not
prove the policy is learning. A rising reward does not by itself prove
that the final command-tracking acceptance criteria pass.

For this project, `learning_curve.csv` turned out to be the best way to
monitor real progress while notebook stdout was silent.

## The `full` 200M run completed successfully

The clean `full` run eventually completed without another code
modification.

Like the successful `t4_fast` run, there was no useful visible stream of
evaluation-step output in the notebook. The final evaluation result
appeared after the long-running training process completed.

The final result was:

``` text
walking_passes: True
```

Every tested command had:

``` text
fall_rate = 0.0
mean_episode_length = 1000.0
```

### Forward command `[0.5, 0.0, 0.0]`

``` text
mean_abs_error = [0.0547553534, 0.0411022240, 0.0779336639]
mean_abs_vy = 0.0411022231
mean_speed = 0.5356212854
fall_rate = 0.0
mean_episode_length = 1000.0
```

### Forward command `[1.0, 0.0, 0.0]`

``` text
mean_abs_error = [0.0576045598, 0.0440581598, 0.0726744248]
mean_abs_vy = 0.0440581590
mean_speed = 1.0070776939
fall_rate = 0.0
mean_episode_length = 1000.0
```

### Lateral command `[0.0, 0.5, 0.0]`

``` text
mean_abs_error = [0.0344475685, 0.0439800688, 0.0603335218]
mean_abs_vy = 0.4859009385
mean_speed = 0.4881531298
fall_rate = 0.0
mean_episode_length = 1000.0
```

### Yaw command `[0.0, 0.0, 1.0]`

``` text
mean_abs_error = [0.0489460084, 0.0364593883, 0.0609441786]
mean_abs_vy = 0.0364593863
mean_speed = 0.0668845475
fall_rate = 0.0
mean_episode_length = 1000.0
```

### Stand command `[0.0, 0.0, 0.0]`

``` text
mean_abs_error = [0.0360154128, 0.0365374747, 0.0953315704]
mean_abs_vy = 0.0365374759
mean_speed = 0.0571563952
fall_rate = 0.0
mean_episode_length = 1000.0
```

## Comparing the passing 30M and 200M policies

Both policies passed, but the 200M policy generally showed tighter
command tracking.

  ---------------------------------------------------------------------------------
  Command                 `t4_fast` error              `full` error `(vx, vy, yaw)`
                          `(vx, vy, yaw)`              
  ----------------------- ---------------------------- ----------------------------
  Forward 0.5             `(0.0646, 0.0442, 0.0974)`   `(0.0548, 0.0411, 0.0779)`

  Forward 1.0             `(0.0823, 0.0596, 0.1065)`   `(0.0576, 0.0441, 0.0727)`

  Lateral 0.5             `(0.0486, 0.0816, 0.0894)`   `(0.0344, 0.0440, 0.0603)`

  Yaw 1.0                 `(0.0502, 0.0582, 0.0995)`   `(0.0489, 0.0365, 0.0609)`

  Stand                   `(0.0413, 0.0403, 0.0931)`   `(0.0360, 0.0365, 0.0953)`
  ---------------------------------------------------------------------------------

The `full` policy also tracked the requested translational commands
closely:

``` text
requested vx = 0.5 m/s → mean speed ≈ 0.536 m/s
requested vx = 1.0 m/s → mean speed ≈ 1.007 m/s
requested vy = 0.5 m/s → mean |vy| ≈ 0.486 m/s
stand command           → mean speed ≈ 0.057 m/s
```

The user's own 200M `full` policy is therefore the preferred final
policy artifact.

## Revised final postmortem

After the controlled `full` experiment:

``` text
30M t4_fast → completed → walking_passes=true
200M full    → completed → walking_passes=true
```

This makes a fundamentally broken PPO implementation, Pup environment,
wrapper, domain-randomization function, or intended training
configuration extremely unlikely as the explanation for the original
problem.

The unresolved issue is specifically why the earlier Colab attempts
behaved differently in wall-clock/runtime behavior.

The exact cause cannot be reconstructed because the earlier attempts
were not instrumented sufficiently before they were stopped.

The strongest practical lesson is:

> **When a JAX/Brax training cell is silent, inspect independent
> evidence of progress before changing code or killing the process.**

Most importantly:

> **Both successful Colab runs could appear silent in the notebook. The
> absence of printed evaluation lines did not mean evaluations were not
> occurring. The records were visible in `learning_curve.csv`, and the
> final result appeared after the training subprocess completed.**

## Updated final state

``` text
Stage 4 wiring:                 PASS
CPU smoke test:                PASS
short GPU/debug PPO test:      PASS
30M t4_fast training:          PASS
30M walking_passes:            TRUE
200M full training:            PASS
200M walking_passes:           TRUE
full fall rate:                0% on all evaluation commands
full episode length:           1000 on all evaluation commands
reference policy required:     NO
preferred final policy:        user's 200M full policy
```
