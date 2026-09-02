from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_iha.interfaces.drone_models import DroneState, LocalPosition


class MissionVideoRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_video_recorder_node")
        self.declare_parameter("image_topic", "/camera")
        self.declare_parameter("mission_state_topic", "/mission/state")
        self.declare_parameter("mission_event_topic", "/mission/event")
        self.declare_parameter("raw_detections_topic", "/perception/raw_detections")
        self.declare_parameter("fusion_topic", "/fusion/target")
        self.declare_parameter("drone_state_topic", "/drone/state")
        self.declare_parameter("local_position_topic", "/drone/local_position")
        self.declare_parameter("output_dir", "/tmp")
        self.declare_parameter("output_prefix", "teknofest_mission")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("overlay_mode", "fusion")
        self.declare_parameter("draw_detection_overlay", True)
        self.declare_parameter("stop_on_landed", True)
        self.declare_parameter("landed_altitude_m", 0.35)
        self.declare_parameter("mission_complete_grace_s", 5.0)
        self.declare_parameter("max_duration_s", 0.0)

        self.bridge = CvBridge()
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None
        self.frame_size: tuple[int, int] | None = None
        self.frame_count = 0
        self.started_at = time.time()
        self.recording_started_at: float | None = None
        self.mission_state: dict = {}
        self.last_event: dict = {}
        self.raw_detections: dict = {}
        self.fusion_packet: dict = {}
        self.drone_state = DroneState()
        self.local_position = LocalPosition()
        self.mission_complete_at: float | None = None
        self.last_video_frame = None

        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self.on_image, 10)
        self.create_subscription(String, str(self.get_parameter("mission_state_topic").value), self.on_mission_state, 10)
        self.create_subscription(String, str(self.get_parameter("mission_event_topic").value), self.on_mission_event, 10)
        self.create_subscription(String, str(self.get_parameter("raw_detections_topic").value), self.on_raw_detections, 10)
        self.create_subscription(String, str(self.get_parameter("fusion_topic").value), self.on_fusion, 10)
        self.create_subscription(String, str(self.get_parameter("drone_state_topic").value), self.on_drone_state, 10)
        self.create_subscription(String, str(self.get_parameter("local_position_topic").value), self.on_local_position, 10)
        self.create_timer(0.5, self.on_timer)
        writer_rate = max(1.0, float(self.get_parameter("fps").value))
        self.create_timer(1.0 / writer_rate, self.on_write_timer)

    def on_mission_state(self, msg: String) -> None:
        self.mission_state = self._json_or_empty(msg.data)
        if self.mission_state.get("state") == "MISSION_COMPLETE" and self.mission_complete_at is None:
            self.mission_complete_at = time.time()

    def on_mission_event(self, msg: String) -> None:
        event = self._json_or_empty(msg.data)
        if event.get("command") == "drop_payload":
            self.last_event = event

    def on_raw_detections(self, msg: String) -> None:
        self.raw_detections = self._json_or_empty(msg.data)

    def on_fusion(self, msg: String) -> None:
        self.fusion_packet = self._json_or_empty(msg.data)

    def on_drone_state(self, msg: String) -> None:
        self.drone_state = DroneState.from_json(msg.data)

    def on_local_position(self, msg: String) -> None:
        self.local_position = LocalPosition.from_json(msg.data)

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.writer is None:
            self._open_writer(frame.shape[1], frame.shape[0])
        if bool(self.get_parameter("draw_detection_overlay").value) and self._overlay_mode() != "none":
            self._draw_detection_overlay(frame)
        self._draw_overlay(frame)
        self.last_video_frame = frame.copy()

    def on_write_timer(self) -> None:
        if self.writer is None or self.last_video_frame is None:
            return
        self.writer.write(self.last_video_frame)
        self.frame_count += 1

    def on_timer(self) -> None:
        if self.writer is None:
            return
        if self._should_stop():
            self.get_logger().info(f"Mission video complete: {self.output_path} ({self.frame_count} frames)")
            self._close_writer()
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self._close_writer()
        return super().destroy_node()

    def _open_writer(self, width: int, height: int) -> None:
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = str(self.get_parameter("output_prefix").value)
        self.output_path = output_dir / f"{prefix}_{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.get_parameter("fps").value)
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.output_path}")
        self.frame_size = (width, height)
        self.recording_started_at = time.time()
        self.get_logger().info(f"Recording mission video: {self.output_path}")

    def _close_writer(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _should_stop(self) -> bool:
        max_duration_s = float(self.get_parameter("max_duration_s").value)
        recording_started_at = self.recording_started_at or self.started_at
        if max_duration_s > 0.0 and time.time() - recording_started_at >= max_duration_s:
            return True
        if not bool(self.get_parameter("stop_on_landed").value):
            return False
        if self.mission_complete_at is None:
            return False
        grace = float(self.get_parameter("mission_complete_grace_s").value)
        if time.time() - self.mission_complete_at < grace:
            return False
        altitude = max(0.0, -self.local_position.z) if self.local_position.z < -0.05 else 0.0
        landed_altitude = float(self.get_parameter("landed_altitude_m").value)
        landed_by_altitude = altitude <= landed_altitude and abs(self.local_position.vz) < 0.2
        return (not self.drone_state.armed) or landed_by_altitude

    def _draw_overlay(self, frame) -> None:
        lines = self._overlay_lines()
        line_height = 22
        margin = 8
        width = max(420, min(frame.shape[1] - 2 * margin, 620))
        height = margin * 2 + line_height * len(lines)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            y = margin + 16 + i * line_height
            cv2.putText(frame, text, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    def _overlay_lines(self) -> list[str]:
        state = self.mission_state.get("state", "UNKNOWN")
        active = self.mission_state.get("active_target", "-")
        released = ",".join(self.mission_state.get("released_targets", []) or [])
        nav_x = float(self.mission_state.get("nav_x", 0.0))
        nav_y = float(self.mission_state.get("nav_y", 0.0))
        fusion_state = self.fusion_packet.get("state", "-")
        selected = self.fusion_packet.get("selected") or {}
        selected_state = selected.get("target_state", selected.get("state", "-")) if selected else "-"
        release_gate = bool(selected.get("release_gate", selected.get("drop_ready", False))) if selected else False
        drop_ready = release_gate or selected_state == "DROP_READY" or state == "DROP_TARGET"
        lines = [
            f"state={state} active={active} fusion={fusion_state}/{selected_state} drop_ready={drop_ready}",
            f"nav=({nav_x:.2f}, {nav_y:.2f}) alt={max(0.0, -self.local_position.z):.2f}m armed={self.drone_state.armed}",
            f"released=[{released}] overlay={self._overlay_mode()}",
        ]
        estimate = self.last_event.get("drop_estimate", {})
        if estimate:
            target = self.last_event.get("target_type", "-")
            distance = float(estimate.get("distance_to_center_m", 0.0))
            inside = bool(estimate.get("inside_target_footprint", False))
            impact = estimate.get("estimated_impact_nav", {})
            lines.append(
                f"last_drop={target} error={distance:.2f}m inside={inside} impact=({float(impact.get('x', 0.0)):.2f},{float(impact.get('y', 0.0)):.2f})"
            )
        else:
            lines.append("last_drop=-")
        return lines

    def _overlay_mode(self) -> str:
        mode = str(self.get_parameter("overlay_mode").value).strip().lower()
        if mode not in {"fusion", "debug", "none"}:
            return "fusion"
        return mode

    def _draw_detection_overlay(self, frame) -> None:
        mode = self._overlay_mode()
        if mode == "debug":
            cv_dets = list(self.raw_detections.get("opencv", []) or [])
            yolo_dets = list(self.raw_detections.get("yolo", []) or [])
            for det in cv_dets:
                self._draw_detection(frame, det, (0, 210, 0), "CV", 2)
            for det in yolo_dets:
                self._draw_detection(frame, det, (255, 0, 255), "YOLO", 2)

        target_label = "FUS" if mode == "debug" else "TARGET"
        selected_label = "FUS SELECT" if mode == "debug" else "LOCKED"
        for det in list(self.fusion_packet.get("targets", []) or []):
            self._draw_detection(frame, det, (0, 180, 255), target_label, 1)
        selected = self.fusion_packet.get("selected")
        if selected:
            self._draw_detection(frame, selected, (0, 255, 255), selected_label, 3)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    def _draw_detection(self, frame, det: dict, color: tuple[int, int, int], source_label: str, thickness: int) -> None:
        bbox = det.get("bbox")
        center = det.get("center")
        if not bbox or len(bbox) != 4:
            return
        x, y, w, h = [int(float(v)) for v in bbox]
        x = max(0, min(frame.shape[1] - 1, x))
        y = max(0, min(frame.shape[0] - 1, y))
        w = max(1, min(frame.shape[1] - x, w))
        h = max(1, min(frame.shape[0] - y, h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if center and len(center) == 2:
            cx, cy = [int(float(v)) for v in center]
            if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                cv2.circle(frame, (cx, cy), 4, color, -1)
        state = det.get("target_state", det.get("state", ""))
        target_type = det.get("target_type", "target")
        detection_confidence = float(det.get("confidence", 0.0))
        fusion_confidence = det.get("fusion_confidence")
        if fusion_confidence is None:
            label = f"{source_label} {target_type} {state} det={detection_confidence:.2f}".strip()
        else:
            label = f"{source_label} {target_type} {state} det={detection_confidence:.2f} fus={float(fusion_confidence):.2f}".strip()
        label_y = max(18, y - 6)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _json_or_empty(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


def main() -> None:
    rclpy.init()
    node = MissionVideoRecorderNode()
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
