from __future__ import annotations

"""ROS 2 fusion process.

The node converts `/perception/raw_detections` into `/fusion/target`.
It owns no camera code and sends no vehicle commands. Its only job is to expose
target confidence, target state, YOLO verification, and release_gate as a clear
contract for the mission manager.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_iha.adapters.fusion_adapter import FusionAdapter
from teknofest_iha.interfaces.detection_models import RawDetectionPacket


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")
        self.declare_parameter("raw_detections_topic", "/perception/raw_detections")
        self.declare_parameter("target_topic", "/fusion/target")
        self.declare_parameter("status_topic", "/fusion/status")
        self.declare_parameter("primary_target", "blue_square")
        self.declare_parameter("secondary_targets", ["red_square"])
        self.adapter = FusionAdapter(
            str(self.get_parameter("primary_target").value),
            list(self.get_parameter("secondary_targets").value),
        )
        self.target_pub = self.create_publisher(String, str(self.get_parameter("target_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(String, str(self.get_parameter("raw_detections_topic").value), self.on_raw, 10)

    def on_raw(self, msg: String) -> None:
        try:
            packet = RawDetectionPacket.from_json(msg.data)
            fused = self.adapter.fuse_packet(packet)
        except Exception as exc:
            self.get_logger().error(f"Fusion failed: {exc}")
            self.status_pub.publish(String(data=f'{{"status":"ERROR","error":"{exc}"}}'))
            return
        self.target_pub.publish(String(data=fused.to_json()))
        self.status_pub.publish(String(data='{"status":"OK"}'))


def main() -> None:
    rclpy.init()
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
