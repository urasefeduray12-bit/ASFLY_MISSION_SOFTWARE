from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("teknofest_iha"), "config", "perception.yaml"])
    return LaunchDescription(
        [
            Node(
                package="teknofest_iha",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="teknofest_iha",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[PathJoinSubstitution([FindPackageShare("teknofest_iha"), "config", "fusion.yaml"])],
            ),
        ]
    )
