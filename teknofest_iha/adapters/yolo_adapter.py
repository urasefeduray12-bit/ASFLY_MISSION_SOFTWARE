from __future__ import annotations

import time
from pathlib import Path

from vision.yolo_detector import YoloAsyncDetector


class YoloAdapter:
    """Adapter around the existing async YOLO detector."""

    def __init__(self, model_path: str, imgsz: int, conf: float, device: str) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.detector = YoloAsyncDetector(self.model_path, imgsz=imgsz, conf=conf, device=device)
        self.started = False
        self.last_result: dict | None = None
        self.last_meta: dict | None = None

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        cleaned = str(model_path).strip().strip("\"'")
        path = Path(cleaned)
        if path.is_absolute() and path.exists():
            return path
        candidates = [
            Path.cwd() / path,
            Path.cwd() / "models_archive" / path.name,
            Path.cwd() / "code" / path.name,
            Path.cwd() / "models_archive" / "iha_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path.cwd() / path

    def start(self) -> None:
        if not self.started:
            self.detector.start()
            self.started = True

    def stop(self) -> None:
        if self.started:
            self.detector.stop()
            self.started = False

    def poll_latest(self) -> dict | None:
        result = self.detector.get_latest_result()
        if result is not None:
            self.last_result = result
            self.last_meta = result.get("meta")
        return self.last_result

    def submit(self, frame_bgr, frame_id: int, roi_center=None, roi_bbox=None) -> dict:
        self.last_meta = self.detector.submit(
            frame_bgr,
            frame_id=frame_id,
            timestamp=time.time(),
            roi_center=roi_center,
            roi_bbox=roi_bbox,
        )
        return self.last_meta
