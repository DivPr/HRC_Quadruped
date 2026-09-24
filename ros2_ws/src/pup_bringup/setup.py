from glob import glob

from setuptools import find_packages, setup

package_name = "pup_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Purdue HRC Software",
    maintainer_email="htsay@purdue.edu",
    description="Policy node and sim2sim bringup for the Pup quadruped.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = pup_bringup.policy_node:main",
        ],
    },
)
