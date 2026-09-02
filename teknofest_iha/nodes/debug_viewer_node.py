from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DebugViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("debug_viewer_node")
        self.declare_parameter("debug_image_topic", "/perception/debug_image")
        self.declare_parameter("window_name", "teknofest_iha_debug")
        self.declare_parameter("display_rate_hz", 20.0)
        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        self.last_image = None
        self.create_subscription(Image, str(self.get_parameter("debug_image_topic").value), self.on_image, 10)
        display_rate = max(1.0, float(self.get_parameter("display_rate_hz").value))
        self.create_timer(1.0 / display_rate, self.on_timer)

    def on_image(self, msg: Image) -> None:
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def on_timer(self) -> None:
        if self.last_image is None:
            return
        cv2.imshow(self.window_name, self.last_image)
        cv2.waitKey(1)

    def destroy_node(self) -> bool:
        cv2.destroyWindow(self.window_name)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DebugViewerNode()
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
