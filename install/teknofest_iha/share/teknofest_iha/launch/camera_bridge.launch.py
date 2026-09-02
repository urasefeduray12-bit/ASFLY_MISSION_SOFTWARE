from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


GZ_CAMERA_TOPIC = "/downward_camera/image"
# /camera/raw is the decision stream. /camera is repeated only for display/recording.


def generate_launch_description():
    return LaunchDescription(
        [
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
        ]
    )
