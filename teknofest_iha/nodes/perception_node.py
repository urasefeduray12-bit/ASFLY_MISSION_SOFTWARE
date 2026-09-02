from __future__ import annotations

"""ROS 2 perception process.

This node is intentionally limited to camera-to-detection work:

* read the decision camera stream,
* run OpenCV color/shape detection,
* schedule YOLO shape verification,
* publish a raw detection packet.

It does not choose mission targets, command the vehicle, or release payloads.
Those responsibilities belong to fusion and mission manager processes.
"""

import json
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_iha.adapters.fusion_adapter import FusionAdapter
from teknofest_iha.adapters.opencv_adapter import OpenCVAdapter
from teknofest_iha.adapters.yolo_adapter import YoloAdapter
from teknofest_iha.interfaces.detection_models import RawDetectionPacket


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("camera_topic", "/camera")
        self.declare_parameter("raw_detections_topic", "/perception/raw_detections")
        self.declare_parameter("fusion_target_topic", "/fusion/target")
        self.declare_parameter("debug_image_topic", "/perception/debug_image")
        self.declare_parameter("status_topic", "/perception/status")
        self.declare_parameter("model_path", "models_archive/iha_best.pt")
        self.declare_parameter("yolo_imgsz", 320)
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_device", "cpu")
        self.declare_parameter("primary_target", "blue_square")
        self.declare_parameter("detect_targets", ["blue_square", "red_square"])
        self.declare_parameter("publish_debug_image", True)

        targets = list(self.get_parameter("detect_targets").value)
        model_path = str(self.get_parameter("model_path").value)
        self.bridge = CvBridge()
        self.frame_id = 0
        self.last_yolo_submit_frame = -10**9
        self.last_yolo_state = "SEARCH"
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)

        self.opencv = OpenCVAdapter(enabled_targets=targets)
        self.yolo = YoloAdapter(
            model_path=model_path,
            imgsz=int(self.get_parameter("yolo_imgsz").value),
            conf=float(self.get_parameter("yolo_conf").value),
            device=str(self.get_parameter("yolo_device").value),
        )
        self.yolo.start()
        self.fusion_policy = FusionAdapter(str(self.get_parameter("primary_target").value), [])

        self.raw_pub = self.create_publisher(String, str(self.get_parameter("raw_detections_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.debug_pub = self.create_publisher(Image, str(self.get_parameter("debug_image_topic").value), 10)
        self.create_subscription(Image, str(self.get_parameter("camera_topic").value), self.on_image, 10)
        self.create_subscription(String, str(self.get_parameter("fusion_target_topic").value), self.on_fusion_target, 10)
        self.get_logger().info(f"Perception using YOLO model: {self.yolo.model_path}")

    def destroy_node(self) -> bool:
        self.yolo.stop()
        return super().destroy_node()

    def on_image(self, msg: Image) -> None:
        self.frame_id += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        debug_image, opencv_dets = self.opencv.detect(frame, self.frame_id)

        yolo_result = self.yolo.poll_latest()
        recent_yolo = self.fusion_policy.yolo_is_fresh(yolo_result, self.frame_id)
        yolo_dets = yolo_result["detections"] if yolo_result and recent_yolo else []
        yolo_error = yolo_result.get("error") if yolo_result else None
        yolo_frame_id = int(yolo_result["frame_id"]) if yolo_result is not None else None
        yolo_age_frames = self.frame_id - yolo_frame_id if yolo_frame_id is not None else None

        cv_target = self._best_cv_target(opencv_dets)
        yolo_every = self.fusion_policy.yolo_interval(self.last_yolo_state)
        if self.frame_id - self.last_yolo_submit_frame >= max(1, yolo_every):
            self.yolo.submit(
                frame.copy(),
                self.frame_id,
                roi_center=cv_target.get("center") if cv_target else None,
                roi_bbox=cv_target.get("bbox") if cv_target else None,
            )
            self.last_yolo_submit_frame = self.frame_id

        packet = RawDetectionPacket(
            frame_id=self.frame_id,
            timestamp=time.time(),
            opencv=opencv_dets,
            yolo=yolo_dets,
            yolo_ran=bool(recent_yolo),
            yolo_frame_id=yolo_frame_id,
            yolo_age_frames=yolo_age_frames,
            yolo_meta=self.yolo.last_meta,
            yolo_error=str(yolo_error) if yolo_error else None,
        )
        self.raw_pub.publish(String(data=packet.to_json()))
        self.status_pub.publish(String(data='{"status":"OK"}'))
        if self.publish_debug_image:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8"))

    def on_fusion_target(self, msg: String) -> None:
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        selected = packet.get("selected") or {}
        state = selected.get("target_state") or packet.get("state")
        if state:
            self.last_yolo_state = str(state)

    @staticmethod
    def _best_cv_target(detections: list[dict]) -> dict | None:
        if not detections:
            return None
        return max(detections, key=lambda d: (d.get("state") == "DETECTED", float(d.get("confidence", 0.0))))


def main() -> None:
    rclpy.init()
    node = PerceptionNode()
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
