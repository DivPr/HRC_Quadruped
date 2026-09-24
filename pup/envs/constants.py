"""The single source of truth for names, ordering, and paths.

Import from here rather than retyping. Three separate pieces of software have to
agree on the joint order and the observation layout — the MJX environment, the
exported policy, and the ROS 2 node — and this module is how they do.

Conventions
-----------
* Leg order: ``FL, FR, RL, RR``. Within a leg: ``hip_abd, hip_flex, knee``.
* All joint axes use the right-hand rule with the **same sign on both sides**:
  ``hip_abd`` about +x, ``hip_flex`` and ``knee`` about +y. Only the lateral
  offset of the hip body is mirrored, so a positive ``hip_abd`` swings the left
  and right legs the same way in world space. This is Playground's Go1
  convention; it means a policy never has to learn a left/right sign flip.
* MuJoCo and JAX quaternions are **wxyz**. ROS ``sensor_msgs/Imu`` is **xyzw**.
* Units: metres, radians, rad/s, Nm.

This module imports numpy and nothing else, so the ROS 2 container's system
Python can import it straight from the bind mount.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[1] / "assets"
CHECKPOINTS = REPO_ROOT / "checkpoints"

#: Torque (`<motor>`) scene, for Stage 1 and the ROS 2 sim node.
SCENE = ASSETS / "scene_flat.xml"
#: Position (`<position kp kv>`) scene with MJX solver settings, for Stages 3-4.
MJX_SCENE = ASSETS / "scene_flat_mjx.xml"

LEGS = ("FL", "FR", "RL", "RR")

#: The 12 actuated joints, in `qpos[7:]` / `qvel[6:]` / `ctrl[:]` order.
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEGS
                    for joint in ("hip_abd", "hip_flex", "knee"))

#: The `home` keyframe's joint angles (rad). Actions are offsets from this.
DEFAULT_POSE = np.tile([0.0, 0.9, -1.8], 4)

#: Per-foot contact sensors, in LEGS order. Read these, never `mjx.Data.contact`.
CONTACT_SENSORS = tuple(f"{leg}_foot_contact" for leg in LEGS)

#: Sensor name -> width. `global_linvel` and `global_angvel` are PRIVILEGED:
#: simulator ground truth that no real robot can measure. Use them in rewards
#: and evaluation, never in the observation.
SENSOR_DIMS = {"gyro": 3, "accelerometer": 3, "orientation": 4,
               "global_linvel": 3, "global_angvel": 3, "upvector": 3,
               **dict.fromkeys(CONTACT_SENSORS, 1)}

#: The 45-dim observation layout, byte-for-byte what `_get_obs` builds, what
#: `policy_node._build_observation` must rebuild, and what travels inside the
#: exported `.npz` so a policy and a robot cannot silently disagree.
OBS_LAYOUT = "gyro[3]|gravity_body[3]|command[3]|joint_offset[12]|joint_velocity[12]|last_action[12]"

#: Action scale: motor_targets = DEFAULT_POSE + ACTION_SCALE * action.
#: Kept in step with `config.default_config().action_scale`.
ACTION_SCALE = 0.5
