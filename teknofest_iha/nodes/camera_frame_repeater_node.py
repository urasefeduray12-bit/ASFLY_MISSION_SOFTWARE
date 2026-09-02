from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraFrameRepeaterNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_frame_repeater_node")
        self.declare_parameter("input_topic", "/camera/raw")
        self.declare_parameter("output_topic", "/camera")
        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("max_repeats_per_frame", 3)
        self.declare_parameter("max_stale_seconds", 0.6)

        self.last_msg: Image | None = None
        self.last_received_at = 0.0
        self.repeats_published = 0

        self.pub = self.create_publisher(Image, str(self.get_parameter("output_topic").value), 10)
        self.create_subscription(Image, str(self.get_parameter("input_topic").value), self.on_image, 10)

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info(
            f"Repeating camera frames {self.get_parameter('input_topic').value} -> {self.get_parameter('output_topic').value} at {rate:.1f} Hz"
        )

    def on_image(self, msg: Image) -> None:
        self.last_msg = msg
        self.last_received_at = time.time()
        self.repeats_published = 0
        self._publish_last()

    def on_timer(self) -> None:
        if self.last_msg is None:
            return
        if time.time() - self.last_received_at > float(self.get_parameter("max_stale_seconds").value):
            return
        if self.repeats_published >= int(self.get_parameter("max_repeats_per_frame").value):
            return
        self._publish_last()

    def _publish_last(self) -> None:
        if self.last_msg is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.last_msg.header.frame_id
        msg.height = self.last_msg.height
        msg.width = self.last_msg.width
        msg.encoding = self.last_msg.encoding
        msg.is_bigendian = self.last_msg.is_bigendian
        msg.step = self.last_msg.step
        msg.data = self.last_msg.data
        self.pub.publish(msg)
        self.repeats_published += 1


def main() -> None:
    rclpy.init()
    node = CameraFrameRepeaterNode()
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
