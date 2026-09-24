"""MuJoCo simulation node: stands in for Pup's hardware and its motor drivers.

On the real robot the policy never sends torques. It sends *joint position
targets* to a motor driver board, which closes a PD loop at its own (much
faster) rate. This node reproduces that split exactly:

* physics + PD run here at 250 Hz, using `scene_flat.xml` (torque `<motor>`
  actuators) and the same equation students wrote in Stage 1;
* the policy runs somewhere else at 50 Hz and only publishes
  `pup_interfaces/JointCommand`.

Published topics
    /pup/joint_states          sensor_msgs/JointState   200 Hz
    /pup/imu                   sensor_msgs/Imu          200 Hz
    /pup/ground_truth/odom     nav_msgs/Odometry         50 Hz  (privileged!)

Subscribed topics
    /pup/joint_command         pup_interfaces/JointCommand

Safety behaviour: if no command has arrived for `command_timeout` seconds the
node falls back to holding the `home` keyframe pose. Real robots do this too --
a policy node that crashes must not leave the legs unpowered mid-stride.
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Imu, JointState

from pup.envs.constants import DEFAULT_POSE, JOINT_NAMES, SCENE
from pup_interfaces.msg import JointCommand

SENSOR_PERIOD_S = 1.0 / 200.0
ODOM_PERIOD_S = 1.0 / 50.0


class SimNode(Node):
    """Run MuJoCo in the loop and expose it over the same topics as hardware."""

    def __init__(self) -> None:
        """Declare parameters, load the scene, and start the timers."""
        super().__init__("pup_sim")
        self.declare_parameter("scene_path", str(SCENE))
        self.declare_parameter("headless", True)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("kp", 25.0)
        self.declare_parameter("kd", 0.5)
        self.declare_parameter("torque_limit", 20.0)
        self.declare_parameter("command_timeout", 0.5)

        scene_path = self.get_parameter("scene_path").value
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe("home").id)
        mujoco.mj_forward(self.model, self.data)

        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)
        self.torque_limit = float(self.get_parameter("torque_limit").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.home_pose = np.asarray(DEFAULT_POSE, float)
        self.target = self.home_pose.copy()
        self.target_kp = np.full(12, self.kp)
        self.target_kd = np.full(12, self.kd)
        self.last_command_time: float | None = None
        self.holding = True

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.joint_publisher = self.create_publisher(JointState, "/pup/joint_states", qos)
        self.imu_publisher = self.create_publisher(Imu, "/pup/imu", qos)
        self.odom_publisher = self.create_publisher(Odometry, "/pup/ground_truth/odom", 10)
        self.create_subscription(JointCommand, "/pup/joint_command", self._on_command, 10)

        realtime_factor = max(1e-3, float(self.get_parameter("realtime_factor").value))
        self.create_timer(self.model.opt.timestep / realtime_factor, self._on_physics)
        self.create_timer(SENSOR_PERIOD_S, self._publish_sensors)
        self.create_timer(ODOM_PERIOD_S, self._publish_odometry)

        self.viewer = None
        if not self.get_parameter("headless").value:
            from mujoco import viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self.create_timer(1.0 / 60.0, self._sync_viewer)
        self.get_logger().info(
            f"pup_sim ready: {1 / self.model.opt.timestep:.0f} Hz physics, "
            f"kp={self.kp}, kd={self.kd}, holding home until a command arrives"
        )

    def _on_command(self, message: JointCommand) -> None:
        """Cache the latest joint targets and gains from the policy node."""
        self.target = np.asarray(message.position, float)
        if np.any(np.asarray(message.kp) > 0):
            self.target_kp = np.asarray(message.kp, float)
            self.target_kd = np.asarray(message.kd, float)
        self.last_command_time = time.monotonic()

    def _active_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (target, kp, kd), falling back to the home pose when stale."""
        stale = (self.last_command_time is None
                 or time.monotonic() - self.last_command_time > self.command_timeout)
        if stale:
            if not self.holding:
                self.get_logger().warn(
                    f"no joint command for {self.command_timeout:.2f}s -- holding home pose")
            self.holding = True
            return self.home_pose, np.full(12, self.kp), np.full(12, self.kd)
        if self.holding:
            self.get_logger().info("joint commands flowing -- policy in control")
        self.holding = False
        return self.target, self.target_kp, self.target_kd

    def _on_physics(self) -> None:
        """Close the PD loop at physics rate and advance the simulation one step."""
        target, kp, kd = self._active_target()
        q = self.data.qpos[7:]
        qd = self.data.qvel[6:]
        torque = np.clip(kp * (target - q) - kd * qd, -self.torque_limit, self.torque_limit)
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

    def _publish_sensors(self) -> None:
        """Publish JointState and Imu from the current MuJoCo state."""
        stamp = self.get_clock().now().to_msg()
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(JOINT_NAMES)
        joint_state.position = self.data.qpos[7:].tolist()
        joint_state.velocity = self.data.qvel[6:].tolist()
        joint_state.effort = self.data.actuator_force.tolist()
        self.joint_publisher.publish(joint_state)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu"
        w, x, y, z = self.data.sensor("orientation").data  # MuJoCo is wxyz
        imu.orientation = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
        gyro = self.data.sensor("gyro").data
        imu.angular_velocity = Vector3(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2]))
        acc = self.data.sensor("accelerometer").data
        imu.linear_acceleration = Vector3(x=float(acc[0]), y=float(acc[1]), z=float(acc[2]))
        self.imu_publisher.publish(imu)

    def _publish_odometry(self) -> None:
        """Publish privileged ground-truth odometry; twist is in the body frame."""
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id = "trunk"
        odom.pose.pose.position.x = float(self.data.qpos[0])
        odom.pose.pose.position.y = float(self.data.qpos[1])
        odom.pose.pose.position.z = float(self.data.qpos[2])
        w, x, y, z = self.data.qpos[3:7]
        odom.pose.pose.orientation = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
        world_velocity = np.asarray(self.data.sensor("global_linvel").data, float)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(self.data.qpos[3:7], float))
        body_velocity = rotation.reshape(3, 3).T @ world_velocity
        odom.twist.twist.linear = Vector3(x=float(body_velocity[0]), y=float(body_velocity[1]),
                                          z=float(body_velocity[2]))
        gyro = self.data.sensor("gyro").data
        odom.twist.twist.angular = Vector3(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2]))
        self.odom_publisher.publish(odom)

    def _sync_viewer(self) -> None:
        """Refresh the passive viewer window, shutting down if the user closes it."""
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit(0)
            self.viewer.sync()


def main(argv: list[str] | None = None) -> None:
    """ROS 2 entry point for ``ros2 run pup_sim sim_node``."""
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
