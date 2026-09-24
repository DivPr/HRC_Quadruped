# Joint-space command sent from the policy node to the simulated motor driver.
# `position` targets are absolute joint angles in radians, in the order given by
# pup.envs.constants.JOINT_NAMES: FL/FR/RL/RR x (hip_abd, hip_flex, knee).
# `kp` (Nm/rad) and `kd` (Nm/(rad/s)) are the per-joint gains the driver should
# use; sending them with every command is how real robots retune stiffness
# on the fly.
std_msgs/Header header
float64[12] position
float64[12] kp
float64[12] kd
