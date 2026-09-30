"""Run the exported Pup policy as a ROS 2 node -- the last step before hardware.

The node has exactly one job: turn the robot's sensor topics into the same
45-number observation vector the MJX environment produced in Stage 3, run the
numpy policy on it at 50 Hz, and publish joint position targets.

Everything that can go wrong here is a *bookkeeping* bug, not a maths bug:

* `sensor_msgs/JointState` gives you names alongside values, and nothing
  guarantees they arrive in the policy's order. Reorder by name, every time.
* `sensor_msgs/Imu.orientation` is **xyzw**. MuJoCo and your Stage 2 helpers are
  **wxyz**. Use `quat_wxyz_from_xyzw`.
* The observation is *relative* joint positions (`q - default_pose`) but
  *absolute* joint velocities.
* `last_action` is the action you published last tick, in [-1, 1] -- not the
  joint target in radians.

Subscribed topics
    /pup/joint_states   sensor_msgs/JointState
    /pup/imu            sensor_msgs/Imu
    /cmd_vel            geometry_msgs/Twist   (vx m/s, vy m/s, wz rad/s)

Published topics
    /pup/joint_command  pup_interfaces/JointCommand   50 Hz
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu, JointState

from pup.envs.constants import CHECKPOINTS, DEFAULT_POSE, JOINT_NAMES, OBS_LAYOUT
from pup.policy import NumpyPolicy, gravity_in_body_frame, quat_wxyz_from_xyzw
from pup_interfaces.msg import JointCommand

CONTROL_PERIOD_S = 0.02  # 50 Hz, matching config.ctrl_dt
OBS_SIZE = 45

# Absolute, so the node works from any working directory (colcon runs it from
# wherever you launched it). Override with the `policy_path` parameter.
DEFAULT_POLICY_PATH = str(CHECKPOINTS / "pup_joystick_flat_reference.npz")


class PolicyNode(Node):
    """Bridge ROS sensor topics to the exported numpy policy and back."""

    def __init__(self) -> None:
        """Declare parameters, load the policy, and wire up topics and the timer."""
        super().__init__("pup_policy")
        self.declare_parameter("policy_path", DEFAULT_POLICY_PATH)
        self.declare_parameter("kp", 25.0)
        self.declare_parameter("kd", 0.5)

        policy_path = str(self.get_parameter("policy_path").value)
        self.policy = NumpyPolicy.load(policy_path)

        if self.policy.obs_size != OBS_SIZE:
            raise ValueError(
                f"policy expects {self.policy.obs_size} observations, not {OBS_SIZE}"
            )

        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)
        self.default_pose = np.asarray(DEFAULT_POSE, float)

        self.joint_state: JointState | None = None
        self.imu: Imu | None = None
        self.command = np.zeros(3)
        self.last_action = np.zeros(12)
        self.publish_count = 0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value

        self.create_subscription(
            JointState,
            "/pup/joint_states",
            self._on_joint_state,
            sensor_qos,
        )

        self.create_subscription(
            Imu,
            "/pup/imu",
            self._on_imu,
            sensor_qos,
        )

        self.create_subscription(
            Twist,
            "/cmd_vel",
            self._on_cmd_vel,
            10,
        )

        self.publisher = self.create_publisher(
            JointCommand,
            "/pup/joint_command",
            10,
        )

        self.create_timer(CONTROL_PERIOD_S, self._on_timer)
        self.create_timer(1.0, self._log_rate)

        self.get_logger().info(
            f"loaded {policy_path}; observation layout: {OBS_LAYOUT}"
        )

    def _on_joint_state(self, message: JointState) -> None:
        """Store the most recent JointState (positions rad, velocities rad/s)."""
        # ===== TODO(student): Cache the latest JointState =====
        self.joint_state = message
        # ===== end TODO =====

    def _on_imu(self, message: Imu) -> None:
        """Store the most recent Imu (orientation xyzw, angular velocity rad/s)."""
        # ===== TODO(student): Cache the latest Imu =====
        self.imu = message
        # ===== end TODO =====

    def _on_cmd_vel(self, message: Twist) -> None:
        """Store the latest teleop command as (vx, vy, wz) in m/s, m/s, rad/s."""
        # ===== TODO(student): Cache the latest velocity command =====
        self.command = np.array([
            message.linear.x,
            message.linear.y,
            message.angular.z,
        ])
        # ===== end TODO =====

    def _ordered_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, qd), each (12,), reordered from JointState.name into JOINT_NAMES order."""
        index = {
            name: position
            for position, name in enumerate(self.joint_state.name)
        }

        order = [index[name] for name in JOINT_NAMES]

        return (
            np.asarray(self.joint_state.position, float)[order],
            np.asarray(self.joint_state.velocity, float)[order],
        )

    def _build_observation(self) -> np.ndarray:
        """Return the (45,) observation, identical in order and units to Stage 3.

        Layout: gyro (3, rad/s) | gravity in body frame (3, unit) |
        command (3, m/s m/s rad/s) | q - default_pose (12, rad) |
        qd (12, rad/s) | last_action (12, unitless).
        """
        # ===== TODO(student): Assemble the 45-dimensional observation =====

        # Reorder joint positions and velocities into the exact order expected
        # by the policy.
        q, qd = self._ordered_joint_state()

        # Angular velocity from ROS IMU, already expressed as xyz.
        gyro = np.array([
            self.imu.angular_velocity.x,
            self.imu.angular_velocity.y,
            self.imu.angular_velocity.z,
        ])

        # ROS stores quaternions as xyzw.
        quat_xyzw = np.array([
            self.imu.orientation.x,
            self.imu.orientation.y,
            self.imu.orientation.z,
            self.imu.orientation.w,
        ])

        # Stage 2 / policy math uses wxyz.
        quat_wxyz = quat_wxyz_from_xyzw(quat_xyzw)

        # Express world gravity in the robot's body frame.
        gravity = gravity_in_body_frame(quat_wxyz)

        # Assemble exactly the same observation layout used during training.
        obs = np.concatenate([
            gyro,
            gravity,
            self.command,
            q - self.default_pose,
            qd,
            self.last_action,
        ])

        if obs.shape != (OBS_SIZE,):
            raise ValueError(
                f"observation has shape {obs.shape}, expected ({OBS_SIZE},)"
            )

        return obs

        # ===== end TODO =====

    def _on_timer(self) -> None:
        """Run one 50 Hz control tick: observe, infer, publish."""
        # ===== TODO(student): Wait for sensors, run the policy, publish the command =====

        # The timer can begin firing before the simulator has published its
        # first sensor messages.
        if self.joint_state is None or self.imu is None:
            return

        # Construct the exact observation representation used during training.
        obs = self._build_observation()

        # Run the neural network once. The result is a normalized 12-D action
        # in [-1, 1].
        action = self.policy(obs)

        # Convert the normalized action into absolute joint position targets.
        targets = (
            self.default_pose
            + self.policy.action_scale * action
        )

        # Send the targets and PD gains to the simulated low-level controller.
        message = JointCommand()
        message.position = targets.tolist()
        message.kp = [self.kp] * 12
        message.kd = [self.kd] * 12

        self.publisher.publish(message)

        # The next observation must contain this normalized action, not the
        # target joint angles.
        self.last_action = action

        # Used by _log_rate() to report the achieved policy frequency.
        self.publish_count += 1

        # ===== end TODO =====

    def _log_rate(self) -> None:
        """Report the achieved control rate once a second."""
        if self.joint_state is None or self.imu is None:
            self.get_logger().warn(
                "waiting for /pup/joint_states and /pup/imu ..."
            )
            return

        self.get_logger().info(
            f"{self.publish_count} Hz | "
            f"command={np.round(self.command, 2).tolist()}"
        )

        self.publish_count = 0


def main(argv: list[str] | None = None) -> None:
    """ROS 2 entry point for ``ros2 run pup_bringup policy_node``."""
    rclpy.init(args=argv if argv is not None else sys.argv)

    node = PolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()