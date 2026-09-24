"""Score the sim2sim loop with the same metrics Stage 4 used.

Drives `/cmd_vel` through a fixed command schedule, listens to the privileged
`/pup/ground_truth/odom`, and prints a JSON report shaped like the one
`pup.train.evaluate` produces so the two are directly comparable.

Exits non-zero if the acceptance thresholds are missed, so CI can use it::

    ros2 run pup_sim eval_sim2sim --duration 15 --out results/eval_sim2sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

SCHEDULE = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
SETTLE_FRACTION = 0.4  # ignore this share of each segment while the gait ramps up
THRESHOLDS = {(0.5, 0.0, 0.0): 0.25}


class EvalNode(Node):
    """Publish a command schedule and accumulate body-frame velocity samples."""

    def __init__(self, duration_s: float) -> None:
        """Set up the /cmd_vel publisher, odom subscription and 50 Hz clock."""
        super().__init__("pup_eval_sim2sim")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/pup/ground_truth/odom", self._on_odom, 10)
        self.segment_s = duration_s / len(SCHEDULE)
        self.duration_s = duration_s
        self.elapsed = 0.0
        self.samples: dict[tuple, list] = {command: [] for command in SCHEDULE}
        self.min_height = math.inf
        self.create_timer(0.02, self._on_timer)

    def _current_command(self) -> tuple:
        """Return the command active at the current elapsed time."""
        index = min(int(self.elapsed / self.segment_s), len(SCHEDULE) - 1)
        return SCHEDULE[index]

    def _on_timer(self) -> None:
        """Publish the scheduled command and advance the evaluation clock."""
        vx, vy, wz = self._current_command()
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
        self.publisher.publish(message)
        self.elapsed += 0.02
        if self.elapsed >= self.duration_s:
            raise SystemExit(0)

    def _on_odom(self, message: Odometry) -> None:
        """Record body-frame (vx, vy, wz) once the current segment has settled."""
        command = self._current_command()
        into_segment = self.elapsed % self.segment_s
        self.min_height = min(self.min_height, message.pose.pose.position.z)
        if into_segment < SETTLE_FRACTION * self.segment_s:
            return
        self.samples[command].append((message.twist.twist.linear.x,
                                      message.twist.twist.linear.y,
                                      message.twist.twist.angular.z))

    def report(self) -> dict:
        """Summarize per-command tracking error and whether thresholds were met."""
        rows, passed = [], True
        for command, samples in self.samples.items():
            if not samples:
                rows.append({"command": list(command), "n_samples": 0})
                passed = False
                continue
            count = len(samples)
            mean = [sum(sample[axis] for sample in samples) / count for axis in range(3)]
            error = [abs(mean[axis] - command[axis]) for axis in range(3)]
            speed = sum(math.hypot(*sample[:2]) for sample in samples) / count
            row = {"command": list(command), "n_samples": count,
                   "mean_velocity": mean, "mean_abs_error": error, "mean_speed": speed}
            limit = THRESHOLDS.get(command)
            if limit is not None:
                row["threshold"] = limit
                row["passed"] = max(error) < limit
                passed &= row["passed"]
            rows.append(row)
        fell = self.min_height < 0.12
        return {"duration_s": self.duration_s, "min_trunk_height": self.min_height,
                "fell": fell, "commands": rows, "passed": bool(passed and not fell)}


def main(argv: list[str] | None = None) -> None:
    """ROS 2 entry point; exits non-zero when the sim2sim thresholds are missed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="total seconds")
    parser.add_argument("--out", default="", help="optional path for the JSON report")
    known, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    rclpy.init(args=ros_args)
    node = EvalNode(known.duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    report = node.report()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if known.out:
        Path(known.out).parent.mkdir(parents=True, exist_ok=True)
        Path(known.out).write_text(text + "\n")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
