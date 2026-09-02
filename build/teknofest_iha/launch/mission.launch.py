from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    mission_config = PathJoinSubstitution([FindPackageShare("teknofest_iha"), "config", "mission.yaml"])
    mavlink_config = PathJoinSubstitution([FindPackageShare("teknofest_iha"), "config", "mavlink.yaml"])
    return LaunchDescription(
        [
            Node(
                package="teknofest_iha",
                executable="mavlink_bridge_node",
                name="mavlink_bridge_node",
                output="screen",
                parameters=[mavlink_config],
            ),
            Node(
                package="teknofest_iha",
                executable="safety_monitor_node",
                name="safety_monitor_node",
                output="screen",
                parameters=[mission_config],
            ),
            Node(
                package="teknofest_iha",
                executable="mission_manager_node",
                name="mission_manager_node",
                output="screen",
                parameters=[mission_config],
            ),
        ]
    )
