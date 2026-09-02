from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("teknofest_iha")
    perception_config = LaunchConfiguration("perception_config")
    fusion_config = LaunchConfiguration("fusion_config")
    mission_config = LaunchConfiguration("mission_config")
    mavlink_config = LaunchConfiguration("mavlink_config")

    default_perception_config = PathJoinSubstitution([pkg, "config", "perception.yaml"])
    default_fusion_config = PathJoinSubstitution([pkg, "config", "fusion.yaml"])
    default_mission_config = PathJoinSubstitution([pkg, "config", "mission.yaml"])
    default_mavlink_config = PathJoinSubstitution([pkg, "config", "mavlink_gcs_router.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="false"),
            DeclareLaunchArgument("perception_config", default_value=default_perception_config),
            DeclareLaunchArgument("fusion_config", default_value=default_fusion_config),
            DeclareLaunchArgument("mission_config", default_value=default_mission_config),
            DeclareLaunchArgument("mavlink_config", default_value=default_mavlink_config),
            DeclareLaunchArgument("console_rate_hz", default_value="5.0"),
            Node(
                package="teknofest_iha",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[perception_config],
            ),
            Node(
                package="teknofest_iha",
                executable="fusion_node",
                name="fusion_node",
                output="screen",
                parameters=[fusion_config],
            ),
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
                parameters=[
                    mission_config,
                    {"autostart": ParameterValue(LaunchConfiguration("autostart"), value_type=bool)},
                ],
            ),
            Node(
                package="teknofest_iha",
                executable="mission_console_node",
                name="mission_console_node",
                output="screen",
                parameters=[
                    {
                        "print_rate_hz": ParameterValue(
                            LaunchConfiguration("console_rate_hz"),
                            value_type=float,
                        )
                    }
                ],
            ),
        ]
    )
