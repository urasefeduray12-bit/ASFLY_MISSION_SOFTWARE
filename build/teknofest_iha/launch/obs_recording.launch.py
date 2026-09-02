from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


GZ_CAMERA_TOPIC = "/downward_camera/image"
# /camera/raw is the decision stream. /camera is repeated only for display/recording.


def generate_launch_description():
    pkg = FindPackageShare("teknofest_iha")
    perception_config = PathJoinSubstitution([pkg, "config", "perception.yaml"])
    fusion_config = PathJoinSubstitution([pkg, "config", "fusion.yaml"])
    mission_config = PathJoinSubstitution([pkg, "config", "mission.yaml"])
    mavlink_config = PathJoinSubstitution([pkg, "config", "mavlink.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="false"),
            DeclareLaunchArgument("record_output_dir", default_value="/tmp"),
            DeclareLaunchArgument("record_max_duration_s", default_value="300.0"),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    f"{GZ_CAMERA_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image",
                    "--ros-args",
                    "-r",
                    f"{GZ_CAMERA_TOPIC}:=/camera/raw",
                ],
                name="camera_bridge",
                output="screen",
            ),
            Node(
                package="teknofest_iha",
                executable="camera_frame_repeater_node",
                name="camera_frame_repeater_node",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/camera/raw",
                        "output_topic": "/camera",
                        "publish_rate_hz": 15.0,
                        "max_repeats_per_frame": 3,
                        "max_stale_seconds": 0.6,
                    }
                ],
            ),
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
                executable="mission_video_recorder_node",
                name="mission_video_recorder_node",
                output="screen",
                parameters=[
                    {
                        "output_dir": LaunchConfiguration("record_output_dir"),
                        "image_topic": "/camera",
                        "overlay_mode": "fusion",
                        "stop_on_landed": False,
                        "max_duration_s": ParameterValue(LaunchConfiguration("record_max_duration_s"), value_type=float),
                    }
                ],
            ),
            Node(
                package="teknofest_iha",
                executable="mission_console_node",
                name="mission_console_node",
                output="screen",
                parameters=[{"print_rate_hz": 5.0}],
            ),
        ]
    )
