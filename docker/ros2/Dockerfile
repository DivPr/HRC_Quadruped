# ROS 2 Jazzy Jalisco (LTS, supported to May 2029) on Ubuntu 24.04.
# The repository is bind-mounted at /ws; nothing is COPYed in, so edits on the
# host are visible instantly inside the container.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-pytest \
      python3-numpy \
      python3-colcon-common-extensions \
      ros-jazzy-rviz2 \
      ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-ros2topic \
      xterm \
      libgl1 \
      libegl1 \
      libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 uses the system Python (3.12 on Jazzy), NOT the uv venv. mujoco is the
# only extra wheel the nodes need; --break-system-packages is the documented
# way to add wheels to a Debian-managed interpreter inside a container.
RUN pip3 install --no-cache-dir --break-system-packages "mujoco==3.13.0"

# `pup.policy` and `pup.envs.constants` are imported straight from the mount.
ENV PYTHONPATH=/ws:$PYTHONPATH
ENV MUJOCO_GL=osmesa
ENV RCUTILS_COLORIZED_OUTPUT=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /ws/ros2_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
