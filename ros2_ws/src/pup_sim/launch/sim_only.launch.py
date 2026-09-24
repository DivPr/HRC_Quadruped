"""Launch the simulator on its own -- the template for your sim2sim.launch.py.

Read this file before writing `pup_bringup/launch/sim2sim.launch.py`. It shows
the three pieces every launch file needs: declared arguments, a `Node` action,
and the `LaunchDescription` that ties them together.

    ros2 launch pup_sim sim_only.launch.py headless:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the launch description for a standalone pup_sim node."""
    headless = LaunchConfiguration("headless")
    realtime_factor = LaunchConfiguration("realtime_factor")
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="true",
                              description="run without the MuJoCo viewer window"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0",
                              description=">1 runs the physics faster than wall clock"),
        Node(
            package="pup_sim",
            executable="sim_node",
            name="pup_sim",
            output="screen",
            parameters=[{"headless": headless, "realtime_factor": realtime_factor}],
        ),
    ])
