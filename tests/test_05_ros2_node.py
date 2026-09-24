"""Stage 5 unit tests. Marked `ros`: they need rclpy, so they only run inside
the ROS 2 container. `tests/test_05_ros2_smoke.sh` runs them for you.

These check the part of `policy_node.py` that the launch-and-watch smoke test
cannot isolate: that `_build_observation` produces exactly the Stage 3 vector,
in the right order, from real ROS messages -- including when `JointState.name`
arrives in an order nobody expected.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.ros

rclpy = pytest.importorskip("rclpy", reason="ROS 2 only (run inside docker/ros2)")

from geometry_msgs.msg import Quaternion, Twist, Vector3  # noqa: E402
from sensor_msgs.msg import Imu, JointState  # noqa: E402

from pup.envs.constants import CHECKPOINTS, DEFAULT_POSE, JOINT_NAMES  # noqa: E402


@pytest.fixture(scope="module")
def ros_context():
    """Initialize rclpy exactly once for this module; it refuses a second init."""
    rclpy.init(args=[])
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros_context):
    """A PolicyNode backed by the reference checkpoint, outside any executor."""
    from pup_bringup.policy_node import PolicyNode

    assert (CHECKPOINTS / "pup_joystick_flat_reference.npz").exists()
    instance = PolicyNode()
    yield instance
    instance.destroy_node()


def _joint_state(positions, velocities, names):
    message = JointState()
    message.name = list(names)
    message.position = [float(v) for v in positions]
    message.velocity = [float(v) for v in velocities]
    return message


def _imu(quaternion_wxyz=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
    message = Imu()
    w, x, y, z = quaternion_wxyz
    message.orientation = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
    message.angular_velocity = Vector3(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2]))
    return message


def test_observation_layout(node):
    """The 45 numbers must land in exactly the Stage 3 slots."""
    q = np.asarray(DEFAULT_POSE) + 0.01 * np.arange(12)
    qd = 0.1 * np.arange(12)
    node._on_joint_state(_joint_state(q, qd, JOINT_NAMES))
    node._on_imu(_imu(gyro=(0.1, -0.2, 0.3)))
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.angular.z = 0.5, -0.25, 1.0
    node._on_cmd_vel(twist)
    node.last_action = np.full(12, 0.5)

    obs = node._build_observation()
    assert obs.shape == (45,)
    np.testing.assert_allclose(obs[0:3], [0.1, -0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(obs[6:9], [0.5, -0.25, 1.0], atol=1e-9)
    np.testing.assert_allclose(obs[9:21], q - np.asarray(DEFAULT_POSE), atol=1e-9)
    np.testing.assert_allclose(obs[21:33], qd, atol=1e-9)
    np.testing.assert_allclose(obs[33:45], 0.5, atol=1e-9)


def test_observation_is_reordered_by_name(node):
    """A driver publishing JointState in a different order must not change the obs."""
    q = np.asarray(DEFAULT_POSE) + 0.01 * np.arange(12)
    qd = 0.1 * np.arange(12)
    node._on_imu(_imu())
    node._on_cmd_vel(Twist())
    node._on_joint_state(_joint_state(q, qd, JOINT_NAMES))
    in_order = node._build_observation()

    order = list(reversed(range(12)))
    node._on_joint_state(_joint_state(q[order], qd[order],
                                      [JOINT_NAMES[i] for i in order]))
    scrambled = node._build_observation()
    np.testing.assert_allclose(in_order, scrambled, atol=1e-12)


def test_imu_quaternion_is_xyzw(node):
    """sensor_msgs/Imu is xyzw; a 90 deg roll must tip gravity onto -y."""
    node._on_joint_state(_joint_state(DEFAULT_POSE, np.zeros(12), JOINT_NAMES))
    node._on_cmd_vel(Twist())
    half = np.sqrt(0.5)
    node._on_imu(_imu(quaternion_wxyz=(half, half, 0.0, 0.0)))  # +90 deg about x
    gravity = node._build_observation()[3:6]
    np.testing.assert_allclose(gravity, [0.0, -1.0, 0.0], atol=1e-6)


def test_timer_publishes_joint_command(node):
    """One tick must publish targets that are default_pose + action_scale * action."""
    published = []
    node.publisher.publish = published.append
    node._on_joint_state(_joint_state(DEFAULT_POSE, np.zeros(12), JOINT_NAMES))
    node._on_imu(_imu())
    node._on_cmd_vel(Twist())
    node.last_action = np.zeros(12)

    node._on_timer()
    assert len(published) == 1
    command = published[0]
    expected = (np.asarray(DEFAULT_POSE)
                + node.policy.action_scale * np.asarray(node.last_action))
    np.testing.assert_allclose(command.position, expected, atol=1e-9)
    assert len(command.kp) == len(command.kd) == 12
    assert all(gain > 0 for gain in command.kp)


def test_timer_waits_for_sensors(node):
    """Before any JointState or Imu arrives, the node must publish nothing."""
    published = []
    node.publisher.publish = published.append
    node._on_timer()
    assert published == []
