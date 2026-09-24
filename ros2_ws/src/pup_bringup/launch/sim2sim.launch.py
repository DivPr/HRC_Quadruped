"""Bring up the whole sim2sim stack: simulator + policy (+ optional teleop).

    ros2 launch pup_bringup sim2sim.launch.py headless:=false teleop:=true
    ros2 launch pup_bringup sim2sim.launch.py policy_path:=/ws/runs/colab/policy.npz

`pup_sim/launch/sim_only.launch.py` is the template this is built from.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the launch description for the full sim2sim stack."""
    # ===== TODO(student): Declare the launch arguments and the three nodes =====
    raise NotImplementedError(
        "Stage 5: Declare the launch arguments and the three nodes. See docs/05_ros2_sim2sim.md")
    # ===== end TODO =====
