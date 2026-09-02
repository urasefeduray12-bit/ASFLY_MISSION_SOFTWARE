from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("output_dir", default_value="/tmp"),
            DeclareLaunchArgument("image_topic", default_value="/camera"),
            DeclareLaunchArgument("overlay_mode", default_value="fusion"),
            DeclareLaunchArgument("stop_on_landed", default_value="false"),
            DeclareLaunchArgument("max_duration_s", default_value="300.0"),
            Node(
                package="teknofest_iha",
                executable="mission_video_recorder_node",
                name="mission_video_recorder_node",
                output="screen",
                parameters=[
                    {
                        "output_dir": LaunchConfiguration("output_dir"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "overlay_mode": LaunchConfiguration("overlay_mode"),
                        "stop_on_landed": LaunchConfiguration("stop_on_landed"),
                        "max_duration_s": LaunchConfiguration("max_duration_s"),
                    }
                ],
            ),
        ]
    )
