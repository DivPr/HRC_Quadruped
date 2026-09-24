"""Numpy ports of the Stage 2 quaternion helpers, for the ROS 2 policy node.

MuJoCo and JAX use ``wxyz`` quaternions. ROS's ``sensor_msgs/Imu`` uses
``xyzw``. Converting between them is the single most common Stage 5 bug, so
``quat_wxyz_from_xyzw`` lives here with a loud name.
"""

from __future__ import annotations

import numpy as np

GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (3,) by the unit wxyz quaternion q (4,); returns (3,) in v's units."""
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    temporary = 2.0 * np.cross(q[1:], v)
    return v + q[0] * temporary + np.cross(q[1:], temporary)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Return the inverse (4,) of a unit wxyz quaternion (dimensionless)."""
    return np.asarray(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


def gravity_in_body_frame(q_wb: np.ndarray) -> np.ndarray:
    """Express the world down-direction in the body frame; (4,) wxyz -> (3,) unit."""
    return quat_rotate(quat_inv(q_wb), GRAVITY_DIRECTION)


def quat_wxyz_from_xyzw(q_xyzw) -> np.ndarray:
    """Reorder a ROS ``geometry_msgs/Quaternion``-style (4,) xyzw into wxyz."""
    x, y, z, w = (float(value) for value in q_xyzw)
    return np.array([w, x, y, z])
