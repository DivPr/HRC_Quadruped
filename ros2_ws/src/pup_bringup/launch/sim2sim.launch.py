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
    headless = LaunchConfiguration("headless")
    policy_path = LaunchConfiguration("policy_path")
    teleop = LaunchConfiguration("teleop")
    realtime_factor = LaunchConfiguration("realtime_factor")

    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="true",
    )

    declare_policy_path = DeclareLaunchArgument(
        "policy_path",
        default_value="",
    )

    declare_teleop = DeclareLaunchArgument(
        "teleop",
        default_value="false",
    )

    declare_realtime_factor = DeclareLaunchArgument(
        "realtime_factor",
        default_value="1.0",
    )

    sim_node = Node(
        package="pup_sim",
        executable="sim_node",
        output="screen",
        parameters=[
            {
                "headless": headless,
                "realtime_factor": realtime_factor,
            }
        ],
    )

    policy_node = Node(
        package="pup_bringup",
        executable="policy_node",
        output="screen",
        parameters=[
            {
                "policy_path": policy_path,
            }
        ],
    )

    teleop_node = Node(
        package="pup_sim",
        executable="teleop_node",
        output="screen",
        condition=IfCondition(teleop),
    )

    return LaunchDescription(
        [
            declare_headless,
            declare_policy_path,
            declare_teleop,
            declare_realtime_factor,
            sim_node,
            policy_node,
            teleop_node,
        ]
    )

    # ===== end TODO =====
