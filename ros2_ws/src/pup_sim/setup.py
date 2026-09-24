from setuptools import find_packages, setup

package_name = "pup_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sim_only.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Purdue HRC Software",
    maintainer_email="htsay@purdue.edu",
    description="MuJoCo simulation node for the Pup quadruped.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = pup_sim.sim_node:main",
            "teleop_node = pup_sim.teleop_node:main",
            "eval_sim2sim = pup_sim.eval_sim2sim:main",
        ],
    },
)
