from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_iha.core.coordinate_frame import CoordinateFrameMapper
from teknofest_iha.core.geofence import Geofence
from teknofest_iha.interfaces.drone_models import LocalPosition


class SafetyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_monitor_node")
        self.declare_parameter("enabled", True)
        self.declare_parameter("x_min", 0.0)
        self.declare_parameter("x_max", 100.0)
        self.declare_parameter("y_min", -15.0)
        self.declare_parameter("y_max", 15.0)
        self.declare_parameter("warning_margin_m", 2.0)
        self.declare_parameter("hard_margin_m", 0.5)
        self.declare_parameter("coordinate_frame", "identity")
        self.enabled = bool(self.get_parameter("enabled").value)
        self.frame_mapper = CoordinateFrameMapper(str(self.get_parameter("coordinate_frame").value))
        self.geofence = Geofence(
            float(self.get_parameter("x_min").value),
            float(self.get_parameter("x_max").value),
            float(self.get_parameter("y_min").value),
            float(self.get_parameter("y_max").value),
            float(self.get_parameter("warning_margin_m").value),
            float(self.get_parameter("hard_margin_m").value),
        )
        self.status_pub = self.create_publisher(String, "/safety/status", 10)
        self.create_subscription(String, "/drone/local_position", self.on_position, 10)

    def on_position(self, msg: String) -> None:
        position = LocalPosition.from_json(msg.data)
        nav_x, nav_y = self.frame_mapper.nav_xy_from_local(position.x, position.y)
        level = self.geofence.check(nav_x, nav_y).value if self.enabled else "DISABLED"
        payload = {
            "status": level,
            "x": nav_x,
            "y": nav_y,
            "local_x": position.x,
            "local_y": position.y,
            "coordinate_frame": self.frame_mapper.mode,
            "timestamp": time.time(),
        }
        self.status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = SafetyMonitorNode()
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
