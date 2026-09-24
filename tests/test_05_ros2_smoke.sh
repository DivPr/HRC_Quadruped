#!/usr/bin/env bash
# Stage 5 smoke test. Runs INSIDE the ROS 2 container (docker/ros2/Dockerfile),
# not in the uv venv. CI calls it; you can too:
#
#   docker compose -f docker/compose.yaml run --rm ros2 /ws/tests/test_05_ros2_smoke.sh
#
# It (1) builds the workspace, (2) launches sim2sim headless with whichever
# policy checkpoint exists, (3) checks /pup/joint_command really publishes at
# ~50 Hz, and (4) runs eval_sim2sim.py and requires the (0.5, 0, 0) command to
# be tracked within 0.25 m/s.
#
# If policy_node.py still has unfilled TODO blocks, the student half is reported
# as NOT STARTED and only the provided half (pup_sim on its own) is checked --
# the same convention conftest.py uses for the Python tests, so CI is green on
# the starter branch.
#
# NOTE: no `set -u` -- ROS 2's setup.bash reads unset variables and would abort.
set -eo pipefail

REPO_ROOT=${REPO_ROOT:-/ws}
POLICY_PATH=${POLICY_PATH:-$REPO_ROOT/checkpoints/pup_joystick_flat_reference.npz}
WS="$REPO_ROOT/ros2_ws"
POLICY_NODE="$WS/src/pup_bringup/pup_bringup/policy_node.py"
LAUNCH_FILE="$WS/src/pup_bringup/launch/sim2sim.launch.py"
LOG_DIR=$(mktemp -d)
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

source "/opt/ros/${ROS_DISTRO}/setup.bash"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export RCUTILS_LOGGING_BUFFERED_STREAM=1

echo "== 1/4 colcon build =="
cd "$WS"
colcon build --symlink-install --event-handlers console_direct+
source "$WS/install/setup.bash"

# --- Is the student's half written yet? ---------------------------------
if grep -q "NotImplementedError" "$POLICY_NODE" "$LAUNCH_FILE"; then
  echo
  echo "▷ NOT STARTED: policy_node.py / sim2sim.launch.py still contain"
  echo "  unfilled TODO(student) blocks. Checking the provided half only."
  echo
  echo "== 2/2 pup_sim on its own =="
  ros2 launch pup_sim sim_only.launch.py headless:=true \
    > "$LOG_DIR/sim.log" 2>&1 &
  sleep 6
  timeout 12 ros2 topic hz /pup/joint_states --window 100 \
    > "$LOG_DIR/hz.log" 2>&1 || true
  RATE=$(grep -o 'average rate: [0-9.]*' "$LOG_DIR/hz.log" | tail -1 | awk '{print $3}')
  if [ -z "$RATE" ]; then
    echo "FAIL: /pup/joint_states never published" >&2
    cat "$LOG_DIR/sim.log" >&2
    exit 1
  fi
  # Nominally 200 Hz. The window is wide because this is a wall-clock rclpy
  # timer sharing a single-threaded executor with the 250 Hz physics loop, and a
  # loaded CI runner drags it down; the check is "the sim node is publishing at
  # roughly the right rate", not a precision measurement.
  echo "  /pup/joint_states at $RATE Hz (nominal 200, accept 120-220)"
  python3 -c "import sys; r=float('$RATE'); sys.exit(0 if 120.0 <= r <= 220.0 else 1)" || {
    echo "FAIL: /pup/joint_states rate $RATE Hz outside 120-220 Hz" >&2; exit 1; }
  echo
  echo "PASS (provided code only): finish Stage 5 to run the full smoke test."
  exit 0
fi

echo "== 2/4 policy_node unit tests =="
python3 -m pytest "$REPO_ROOT/tests/test_05_ros2_node.py" -q -m ros -p no:cacheprovider

echo "== 3/4 launch sim2sim (headless) and check the rate =="
if [ ! -f "$POLICY_PATH" ]; then
  echo "FAIL: no policy at $POLICY_PATH (see checkpoints/README.md)" >&2
  exit 1
fi
ros2 launch pup_bringup sim2sim.launch.py \
  headless:=true policy_path:="$POLICY_PATH" > "$LOG_DIR/launch.log" 2>&1 &
LAUNCH_PID=$!
sleep 8

if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
  echo "FAIL: launch exited early" >&2; cat "$LOG_DIR/launch.log" >&2; exit 1
fi

timeout 12 ros2 topic hz /pup/joint_command --window 100 > "$LOG_DIR/hz.log" 2>&1 || true
RATE=$(grep -o 'average rate: [0-9.]*' "$LOG_DIR/hz.log" | tail -1 | awk '{print $3}')
if [ -z "$RATE" ]; then
  echo "FAIL: /pup/joint_command never published" >&2
  cat "$LOG_DIR/launch.log" >&2
  exit 1
fi
echo "measured $RATE Hz (want 45-55)"
python3 -c "import sys; r=float('$RATE'); sys.exit(0 if 45.0 <= r <= 55.0 else 1)" || {
  echo "FAIL: /pup/joint_command rate $RATE Hz outside 45-55 Hz" >&2; exit 1; }

echo "== 4/4 eval_sim2sim =="
ros2 run pup_sim eval_sim2sim --duration 20 --out "$REPO_ROOT/results/eval_sim2sim.json"

echo "PASS: Stage 5 sim2sim smoke test"
