# Stage 5 — ROS 2 sim2sim deployment

## Why this exists

Your policy currently runs inside the same process, on the same arrays, in the
same framework it was trained in. That is the easiest possible situation and it
proves almost nothing about deployment.

**Sim2sim** means running the trained policy against a *different* simulator
process, through the *same interfaces the real robot uses* — ROS 2 topics, at
real-time rate, with sensor messages instead of array slices. It is the last
step before hardware, and it catches the bugs that hardware would otherwise
catch.

## ROS 2 in one page

Skip if you know ROS.

- A **node** is a process that does one job. `sim_node` simulates; `policy_node`
  thinks; `teleop_node` reads your keyboard. They are separate processes that
  find each other over the network.
- A **topic** is a named, typed channel: `/pup/imu` carries `sensor_msgs/Imu`.
  Publishers and subscribers never know about each other, only about the topic.
  This is why you can swap `sim_node` for real hardware without touching
  `policy_node` — and that swap is the entire point of the exercise.
- A **message** is a struct defined in a `.msg` file and code-generated into
  every supported language.
- A **parameter** is a named, typed setting on a node, set from the command line
  or a launch file (`policy_path`, `kp`, `headless`).
- A **launch file** is a Python script that starts several nodes with the right
  parameters, so you type one command instead of four.
- **QoS** is per-topic delivery policy. Sensor streams use "best effort, keep
  last 1" (a stale IMU reading is worse than none); commands use reliable
  delivery. Mismatched QoS between a publisher and subscriber is the classic
  "the topic exists but I receive nothing" bug.

Commands you will actually use:

```bash
ros2 topic list
ros2 topic hz /pup/joint_command      # is your node really running at 50 Hz?
ros2 node list
ros2 param get /pup_policy policy_path
rqt_graph                             # who talks to whom (needs a display)

# Sensor topics here are published best-effort, so a default (reliable)
# subscriber receives nothing at all. This is the QoS trap, live:
ros2 topic echo /pup/joint_states --once --qos-reliability best_effort
```

> **Reading:** the ROS 2 Jazzy beginner CLI tutorial, "Writing a simple
> publisher and subscriber (Python)", "Creating a launch file", and "About
> Quality of Service settings" ([reading packet](reading_packet.md), Stage 5).
> The first one is ninety minutes and covers most of what you need here.

## The architecture

`sim_node` deliberately does **not** know a policy exists. It receives joint
position targets and gains, closes a PD loop at 250 Hz, and publishes sensors.
That is exactly the contract a motor driver board offers. If no command arrives
for 0.5 s it holds the `home` pose.

`/pup/ground_truth/odom` is **privileged**: absolute position and velocity that
no real robot has. It exists so `eval_sim2sim.py` can score you. Using it in
`policy_node` is cheating and will be visible in review.

## Getting the environment

```bash
docker compose -f docker/compose.yaml build          # ~2 GB, once
docker compose -f docker/compose.yaml run --rm ros2  # drops you into /ws/ros2_ws
```

Inside the container:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch pup_bringup sim2sim.launch.py
```

The repository is bind-mounted at `/ws`, so edits on your host are visible
immediately — with `--symlink-install`, Python changes need only a node restart,
not a rebuild. (New *files* and any `setup.py` change do need a rebuild.)

**Two Pythons, again.** The container uses the system Python 3.12 with ROS 2 and
`mujoco` installed by `apt`/`pip`. It does **not** use `.venv`, and it has no
JAX. `PYTHONPATH=/ws` lets it import `pup.policy` and `pup.envs.constants`
directly from the mount — both of those are numpy-only for exactly this reason.

**Already have ROS 2 Jazzy natively?** Then:
```bash
pip install --user mujoco==3.13.0
export PYTHONPATH=$PWD:$PYTHONPATH
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
```

**GUI (viewer, RViz, teleop window):** uncomment the `DISPLAY` lines in
`docker/compose.yaml`. On Linux run `xhost +local:docker` first; on macOS start
XQuartz and enable "Allow connections from network clients". Headless is the
default and everything except the pretty picture works without a display.

## What is provided

- `pup_interfaces` — an `ament_cmake` package containing exactly one message,
  `JointCommand` (`header`, `position[12]`, `kp[12]`, `kd[12]`). **Why a
  separate package?** Because message definitions are code-generated into C++,
  Python and every other client language, and that generation is a CMake job.
  Keeping them in their own dependency-light package means a C++ node can depend
  on your messages without dragging in your Python. Every ROS 2 project you will
  ever see does this.
- `pup_sim` — `sim_node`, `teleop_node`, `eval_sim2sim.py`, and
  `launch/sim_only.launch.py` (the template for your launch file).

## Your task

### 1. Complete `pup_bringup/pup_bringup/policy_node.py`

The node skeleton, parameter declarations, subscriptions, publisher, timer and
`main()` are provided. Three TODO blocks:

**(a) The three callbacks.** Cache the latest `JointState`, `Imu` and `Twist`.
Three or four lines total. Store the `Twist` as an `np.array([linear.x,
linear.y, angular.z])` — that is your `(vx, vy, wz)` command.

**(b) `_build_observation() -> np.ndarray[45]`.** The same 45 numbers, in the
same order, in the same units as Stage 3:

```
gyro(3) | gravity_in_body(3) | command(3) | q - default_pose(12) | qd(12) | last_action(12)
```

Three traps, all of them real:

- **`JointState.name` ordering.** ROS gives you names alongside values and
  guarantees nothing about their order. Build an index from
  `JointState.name` and reorder into `JOINT_NAMES` order. `_ordered_joint_state()`
  is provided — use it, and understand why it exists. (`sim_node` happens to
  publish in the right order today. Depending on that is how you write a node
  that breaks the day someone else's driver publishes it alphabetically.)
- **`sensor_msgs/Imu.orientation` is `xyzw`.** MuJoCo, MJX and your Stage 2
  helper are `wxyz`. Use `quat_wxyz_from_xyzw` from `pup.policy`. Getting this
  wrong gives you a robot whose sense of "down" is rotated, which looks exactly
  like "the policy is bad".
- **`last_action` is the action, not the target.** It is what `NumpyPolicy`
  returned last tick, in `[−1, 1]` — not the joint angles in radians you
  published.

`gravity_in_body_frame` and `quat_wxyz_from_xyzw` are provided in numpy in
`pup/policy/math_numpy.py`; they are line-for-line ports of your Stage 2 JAX
versions.

**(c) `_on_timer()`.** One 50 Hz control tick:
- return early until at least one `JointState` **and** one `Imu` have arrived
  (on startup they have not, and `None.position` is an ugly way to find out);
- build the observation, run the policy;
- publish a `JointCommand` whose `position` is
  `default_pose + policy.action_scale * action`, with `kp`/`kd` from the node
  parameters. (`policy.joint_targets(obs)` computes the same thing, but it runs
  the network a second time — you already have `action`.)
- store `last_action` for the next tick;
- the 1 Hz logging is already wired to `self.publish_count`; increment it.

### 2. Write `pup_bringup/launch/sim2sim.launch.py`

Launch `sim_node` and `policy_node`, plus `teleop_node` conditionally. Declare
launch arguments `headless`, `policy_path`, `teleop`, and `realtime_factor`, and
pass them through as node parameters. `pup_sim/launch/sim_only.launch.py` is the
worked template; the only new ideas are `IfCondition` and passing a
`LaunchConfiguration` into `parameters=[{...}]`.

```bash
ros2 launch pup_bringup sim2sim.launch.py headless:=false teleop:=true
ros2 launch pup_bringup sim2sim.launch.py policy_path:=/ws/runs/colab/policy.npz
```

### 3. Drive it and measure it

```bash
# terminal 1
ros2 launch pup_bringup sim2sim.launch.py headless:=false
# terminal 2 (docker compose ... run --rm ros2 bash)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# terminal 3
ros2 topic hz /pup/joint_command      # want ~50
ros2 run pup_sim eval_sim2sim --duration 20 --out /ws/results/eval_sim2sim.json
```

Record a screen capture GIF of it walking under your keyboard control, and
commit it plus the eval JSON to `results/`.

**A real result to think about.** The reference policy tracks a 1.0 m/s command
almost perfectly through ROS (0.952 m/s measured) but undershoots a 0.5 m/s
command (0.383 m/s) — and in MJX, during Stage 4 evaluation, it hit 0.52 m/s on
that same command. Same weights, same observation, different answer. The
plain-MuJoCo scene solves contacts with `iterations="20"` and closes its PD loop
in numpy at 250 Hz; the MJX scene uses `iterations="1"` and MuJoCo's own
`<position>` actuators. That is a genuine sim2sim gap, found by exactly the
exercise you just ran, and it is a smaller version of what you would see going
to hardware. Look for it in your own numbers.

### 4. Reflection (one paragraph in `SUBMISSION.md`)

**List two ways sim2sim can pass while real hardware still fails, and what you
would add to the sim node to catch each.** Some real ones, to think with rather
than copy: the sim node replies in the same millisecond while a real motor
driver is 5–15 ms behind over CAN; a real IMU has a bias that drifts with
temperature and a real gyro has noise your `noise_config` didn't model; real
motors saturate, and their torque limit falls as they heat up; real encoders
have a fixed offset from your URDF's zero; a real robot's mass is not 11.7 kg
because someone taped a battery to it.

## How you'll know you're done

Inside the container:

```bash
/ws/tests/test_05_ros2_smoke.sh
```

It builds the workspace, launches `sim2sim.launch.py` headless with the
reference checkpoint, asserts `/pup/joint_command` publishes at **45–55 Hz**,
then runs `eval_sim2sim.py` and requires the `(0.5, 0, 0)` command to be tracked
within **0.25 m/s** without the trunk dropping below 0.12 m.

## Common mistakes

- **`colcon build` from the wrong directory.** Run it in `/ws/ros2_ws`, not
  `/ws` and not `src/`.
- **`--symlink-install` didn't pick up my new file.** Symlinks cover *existing*
  Python files. A new module, a new entry point, or any `setup.py` edit needs a
  real rebuild.
- **Nodes cannot see each other.** DDS discovery uses multicast, which does not
  cross Docker's default bridge network. `compose.yaml` sets
  `network_mode: host` for this reason. If you run containers by hand, keep it.
  Also set the same `ROS_DOMAIN_ID` everywhere.
- **The topic exists but `echo` shows nothing.** QoS mismatch. Sensor topics
  here are best-effort; a reliable subscriber will silently receive nothing.
- **Publishing at 250 Hz because you reused the sim's timer.** The policy runs
  at 50 Hz. It was trained at 50 Hz. `action_rate` penalties, `last_action`, and
  the whole dynamics assume 50 Hz.
- **`xyzw`.** Yes, again.
- **Sourcing ROS inside the uv venv.** Don't mix them.
- **Using `/pup/ground_truth/odom` in the policy node.** Cheating, and obvious.

