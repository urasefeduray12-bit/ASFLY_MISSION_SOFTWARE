from setuptools import find_packages, setup


package_name = "teknofest_iha"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*", "vision", "vision.*", "control", "control.*", "utils", "utils.*"]),
    py_modules=["config"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", [
            "config/mission.yaml",
            "config/perception.yaml",
            "config/fusion.yaml",
            "config/mavlink.yaml",
            "config/mavlink_gcs_router.yaml",
        ]),
        (f"share/{package_name}/launch", [
            "launch/perception.launch.py",
            "launch/mission.launch.py",
            "launch/field_mission.launch.py",
            "launch/full_sim.launch.py",
            "launch/camera_bridge.launch.py",
            "launch/record_mission.launch.py",
            "launch/obs_recording.launch.py",
        ]),
        (f"share/{package_name}/docs", [
            "docs/UNIX_MODULAR_ARCHITECTURE.md",
            "docs/TOPICS_AND_CONTRACTS.md",
            "docs/STATE_MACHINES.md",
            "docs/CODE_REFERENCE.md",
            "docs/REAL_UAV_READINESS_CHECKLIST.md",
            "docs/GCS_MAVLINK_INTEGRATION.md",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="omer",
    maintainer_email="omer@example.com",
    description="ROS 2 package for Teknofest UAV mission architecture.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = teknofest_iha.nodes.perception_node:main",
            "fusion_node = teknofest_iha.nodes.fusion_node:main",
            "mission_manager_node = teknofest_iha.nodes.mission_manager_node:main",
            "mavlink_bridge_node = teknofest_iha.nodes.mavlink_bridge_node:main",
            "safety_monitor_node = teknofest_iha.nodes.safety_monitor_node:main",
            "debug_viewer_node = teknofest_iha.nodes.debug_viewer_node:main",
            "mission_video_recorder_node = teknofest_iha.nodes.mission_video_recorder_node:main",
            "camera_frame_repeater_node = teknofest_iha.nodes.camera_frame_repeater_node:main",
            "mission_console_node = teknofest_iha.nodes.mission_console_node:main",
        ],
    },
)
