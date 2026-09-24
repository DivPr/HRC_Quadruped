#!/usr/bin/env bash
# Source the ROS 2 distro, then the workspace overlay if it has been built.
set -e
source /opt/ros/"$ROS_DISTRO"/setup.bash
if [ -f /ws/ros2_ws/install/setup.bash ]; then
  source /ws/ros2_ws/install/setup.bash
fi
exec "$@"
