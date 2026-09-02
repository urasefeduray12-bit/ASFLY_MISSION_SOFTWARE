from __future__ import annotations

"""ROS 2 to MAVLink bridge.

The bridge is a transport boundary. It consumes high-level `/drone/cmd_*`
messages and translates them into MAVLink actions, then publishes telemetry
back into ROS. Mission sequencing must not be added here.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_iha.adapters.mavlink_adapter import MavlinkAdapter
from teknofest_iha.interfaces.drone_models import command_json


class MavlinkBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("mavlink_bridge_node")
        self.declare_parameter("connection", "udp:127.0.0.1:14551")
        self.declare_parameter("heartbeat_timeout_s", 30.0)
        self.declare_parameter("telemetry_rate_hz", 20.0)
        self.declare_parameter("command_timeout_s", 5.0)
        self.declare_parameter("mode", "GUIDED")
        self.declare_parameter("source_system", 255)
        self.declare_parameter("auto_connect", True)

        self.adapter: MavlinkAdapter | None = None
        self.connected = False
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)

        self.state_pub = self.create_publisher(String, "/drone/state", 10)
        self.local_pub = self.create_publisher(String, "/drone/local_position", 10)
        self.altitude_pub = self.create_publisher(String, "/drone/altitude", 10)
        self.status_pub = self.create_publisher(String, "/drone/status", 10)

        self.create_subscription(String, "/drone/cmd_mode", self.on_mode, 10)
        self.create_subscription(String, "/drone/cmd_arm", self.on_arm, 10)
        self.create_subscription(String, "/drone/cmd_takeoff", self.on_takeoff, 10)
        self.create_subscription(String, "/drone/cmd_velocity", self.on_velocity, 10)
        self.create_subscription(String, "/drone/cmd_position", self.on_position, 10)
        self.create_subscription(String, "/drone/cmd_land", self.on_land, 10)
        self.create_subscription(String, "/drone/cmd_drop", self.on_drop, 10)

        rate = float(self.get_parameter("telemetry_rate_hz").value)
        self.create_timer(1.0 / max(1.0, rate), self.on_timer)
        if bool(self.get_parameter("auto_connect").value):
            self.connect()

    def connect(self) -> None:
        if self.connected:
            return
        try:
            self.adapter = MavlinkAdapter(
                str(self.get_parameter("connection").value),
                source_system=int(self.get_parameter("source_system").value),
                heartbeat_timeout_s=float(self.get_parameter("heartbeat_timeout_s").value),
            )
            self.adapter.connect()
            self.connected = True
            self.status_pub.publish(String(data='{"status":"CONNECTED"}'))
        except Exception as exc:
            self.get_logger().error(f"MAVLink connect failed: {exc}")
            self.status_pub.publish(String(data=json.dumps({"status": "ERROR", "error": str(exc)})))

    def on_timer(self) -> None:
        if not self.connected or self.adapter is None:
            return
        try:
            status = self.adapter.read_messages(timeout_s=0.02)
        except Exception as exc:
            self.connected = False
            self.status_pub.publish(String(data=json.dumps({"status": "ERROR", "error": str(exc)})))
            return
        self.state_pub.publish(String(data=status.state.to_json()))
        self.local_pub.publish(String(data=status.local_position.to_json()))
        self.altitude_pub.publish(String(data=status.altitude.to_json()))

    def on_mode(self, msg: String) -> None:
        data = json.loads(msg.data)
        self._call(lambda: self.adapter.set_mode(str(data.get("mode", "GUIDED")), self.command_timeout_s))

    def on_arm(self, msg: String) -> None:
        data = json.loads(msg.data)
        arm = bool(data.get("arm", True))
        if arm:
            self._call(lambda: self.adapter.arm(self.command_timeout_s))
        else:
            self._call(lambda: self.adapter.disarm())

    def on_takeoff(self, msg: String) -> None:
        data = json.loads(msg.data)
        self._call(lambda: self.adapter.takeoff(float(data["altitude_m"])))

    def on_velocity(self, msg: String) -> None:
        data = json.loads(msg.data)
        self._call(lambda: self.adapter.send_velocity_ned(float(data.get("vx", 0.0)), float(data.get("vy", 0.0)), float(data.get("vz", 0.0)), float(data.get("yaw_rate", 0.0))))

    def on_position(self, msg: String) -> None:
        data = json.loads(msg.data)
        self._call(lambda: self.adapter.send_position_ned(float(data["x"]), float(data["y"]), float(data["z"]), data.get("yaw")))

    def on_land(self, msg: String) -> None:
        self._call(lambda: self.adapter.land())

    def on_drop(self, msg: String) -> None:
        data = json.loads(msg.data)
        if bool(data.get("dry_run", True)):
            self.status_pub.publish(String(data=json.dumps({"status": "DROP_DRY_RUN", "target": data.get("target_type")})))
            return
        self._call(lambda: self.adapter.drop_payload(int(data["servo"]), int(data["pwm"]), float(data.get("hold_seconds", 0.8)), data.get("reset_pwm")))

    def _call(self, fn) -> None:
        if self.adapter is None:
            self.connect()
        if self.adapter is None:
            return
        try:
            result = fn()
            self.status_pub.publish(String(data=json.dumps({"status": "OK", "result": result})))
        except Exception as exc:
            self.get_logger().error(f"MAVLink command failed: {exc}")
            self.status_pub.publish(String(data=json.dumps({"status": "ERROR", "error": str(exc)})))


def main() -> None:
    rclpy.init()
    node = MavlinkBridgeNode()
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
