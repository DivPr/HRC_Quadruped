"""Keyboard teleop for Pup.

`teleop_twist_keyboard` already does the hard part, so this node is a thin
convenience wrapper for people who would rather press four arrow keys than
remember its key map. It publishes `geometry_msgs/Twist` on `/cmd_vel`, which
is exactly what the policy node consumes as its (vx, vy, wz) command.

Run either of these -- they are interchangeable::

    ros2 run pup_sim teleop_node
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from __future__ import annotations

import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

KEY_BINDINGS = {
    "w": (0.5, 0.0, 0.0), "s": (-0.5, 0.0, 0.0),
    "a": (0.0, 0.4, 0.0), "d": (0.0, -0.4, 0.0),
    "q": (0.0, 0.0, 0.8), "e": (0.0, 0.0, -0.8),
    " ": (0.0, 0.0, 0.0),
}
BANNER = """Pup teleop
  w/s  forward / backward      a/d  strafe left / right
  q/e  turn left / right       space  stop
  x    quit
Commands are clamped to the training ranges: vx [-1.0, 1.5] m/s,
vy [-0.8, 0.8] m/s, wz [-1.2, 1.2] rad/s.
"""


class TeleopNode(Node):
    """Read single keypresses and republish them as /cmd_vel."""

    def __init__(self) -> None:
        """Create the /cmd_vel publisher and print the key map."""
        super().__init__("pup_teleop")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        print(BANNER, flush=True)

    def publish(self, vx: float, vy: float, wz: float) -> None:
        """Publish one (vx, vy, wz) command in m/s, m/s, rad/s."""
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
        self.publisher.publish(message)


def _read_key() -> str:
    """Block until one character is typed, without waiting for Enter."""
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(argv: list[str] | None = None) -> None:
    """ROS 2 entry point for ``ros2 run pup_sim teleop_node``."""
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = TeleopNode()
    try:
        while rclpy.ok():
            key = _read_key()
            if key in ("x", "\x03"):
                break
            if key in KEY_BINDINGS:
                node.publish(*KEY_BINDINGS[key])
    finally:
        node.publish(0.0, 0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
