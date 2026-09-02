from __future__ import annotations

from typing import Any

from vision.opencv_detector import OpenCVDetector


class OpenCVAdapter:
    """Adapter around the existing OpenCV color/contour detector."""

    def __init__(self, enabled_targets: list[str] | None = None) -> None:
        self.detector = OpenCVDetector()
        self.enabled_targets = set(enabled_targets or [])

    def detect(self, frame_bgr, frame_id: int) -> tuple[Any, list[dict]]:
        debug_image, detections = self.detector.detect(frame_bgr, frame_id=frame_id)
        if self.enabled_targets:
            detections = [d for d in detections if d.get("target_type") in self.enabled_targets]
        return debug_image, detections
