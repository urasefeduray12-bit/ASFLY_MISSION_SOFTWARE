"""Asynchronous YOLO square detector.

The model is treated as a single-class square detector. Outputs such as
`unknown` or `square_unknown` are normalized to `square` so downstream fusion
can use YOLO as shape verification without assigning color authority to it.
"""

import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config
from vision.detection_types import clamp_bbox_xywh, make_detection, xyxy_to_xywh


class YoloAsyncDetector:
    def __init__(self, model_path, imgsz=320, conf=0.25, device="cpu", class_map=None):
        self.model_path = Path(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device
        self.class_map = class_map or config.YOLO_CLASS_MAP
        self.job_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.model = None
        self.load_error = None

    def start(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model bulunamadi: {self.model_path}")
        self.thread = threading.Thread(target=self._worker, name="YoloAsyncDetector", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def submit(self, frame_bgr, frame_id, timestamp, roi_center=None, roi_bbox=None):
        canvas, meta = make_yolo_input(frame_bgr, self.imgsz, roi_center=roi_center, roi_bbox=roi_bbox)
        job = {
            "canvas": canvas,
            "meta": meta,
            "frame_shape": frame_bgr.shape,
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
        }
        self._put_latest(self.job_queue, job)
        return meta

    def get_latest_result(self):
        latest = None
        while True:
            try:
                latest = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    @staticmethod
    def _put_latest(q, item):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _load_model(self):
        ultralytics_dir = Path(
            os.environ.get("ASFLY_ULTRALYTICS_DIR", str(Path(tempfile.gettempdir()) / "asfly_ultralytics"))
        )
        ultralytics_dir.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
        from ultralytics import YOLO

        return YOLO(str(self.model_path))

    def _worker(self):
        try:
            self.model = self._load_model()
        except Exception as exc:
            self.load_error = exc
            self._put_latest(
                self.result_queue,
                {
                    "detections": [],
                    "infer_ms": 0.0,
                    "frame_id": -1,
                    "timestamp": time.time(),
                    "meta": {"mode": "ERROR"},
                    "error": str(exc),
                },
            )
            return

        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                raw = self._run_inference(job["canvas"])
                infer_ms = (time.perf_counter() - t0) * 1000.0
                detections = self._to_standard_detections(raw, job)
                result = {
                    "detections": detections,
                    "infer_ms": infer_ms,
                    "frame_id": job["frame_id"],
                    "timestamp": job["timestamp"],
                    "meta": job["meta"],
                    "error": None,
                }
            except Exception as exc:
                result = {
                    "detections": [],
                    "infer_ms": (time.perf_counter() - t0) * 1000.0,
                    "frame_id": job["frame_id"],
                    "timestamp": job["timestamp"],
                    "meta": job["meta"],
                    "error": str(exc),
                }
            self._put_latest(self.result_queue, result)

    def _run_inference(self, canvas):
        result = self.model.predict(
            canvas,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        dets = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            dets.append((cls_id, score, x1, y1, x2, y2))
        return dets

    def _to_standard_detections(self, raw_dets, job):
        detections = []
        for cls_id, score, x1, y1, x2, y2 in raw_dets:
            fx1, fy1, fx2, fy2 = canvas_box_to_frame(x1, y1, x2, y2, job["meta"])
            bbox = clamp_bbox_xywh(xyxy_to_xywh(fx1, fy1, fx2, fy2), job["frame_shape"])
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            target_type = self._normalize_target_type(self.class_map.get(cls_id, "square"))
            detections.append(
                make_detection(
                    "yolo",
                    target_type,
                    bbox,
                    score,
                    "DETECTED",
                    job["frame_id"],
                    timestamp=job["timestamp"],
                    error=None,
                )
            )
        return detections

    @staticmethod
    def _normalize_target_type(target_type):
        target = str(target_type).strip().lower()
        if target in {"unknown", "square_unknown", "unknown_square"}:
            return "square"
        return target


def make_yolo_input(frame, imgsz, roi_center=None, roi_bbox=None):
    frame_h, frame_w = frame.shape[:2]
    roi_rect = None

    if roi_bbox is not None:
        x, y, w, h = roi_bbox
        cx = x + w / 2
        cy = y + h / 2
        side = max(config.YOLO_MIN_ROI_SIZE, int(max(w, h) * config.YOLO_ROI_SCALE), imgsz)
        roi_rect = _roi_from_center(cx, cy, side, frame_w, frame_h)
    elif roi_center is not None:
        roi_rect = _roi_from_center(roi_center[0], roi_center[1], imgsz, frame_w, frame_h)

    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        scale = imgsz / max(crop_w, crop_h)
        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        pad_x = (imgsz - new_w) // 2
        pad_y = (imgsz - new_h) // 2
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        return canvas, {
            "mode": "ROI",
            "scale": scale,
            "pad": (pad_x, pad_y),
            "offset": (x1, y1),
            "roi_rect": (x1, y1, x2, y2),
        }

    scale = imgsz / max(frame_w, frame_h)
    new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, {
        "mode": "FULL",
        "scale": scale,
        "pad": (pad_x, pad_y),
        "offset": (0, 0),
        "roi_rect": (0, 0, frame_w, frame_h),
    }


def _roi_from_center(cx, cy, side, frame_w, frame_h):
    side = int(max(1, side))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        x1 -= x2 - frame_w
        x2 = frame_w
    if y2 > frame_h:
        y1 -= y2 - frame_h
        y2 = frame_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    return x1, y1, x2, y2


def canvas_box_to_frame(x1, y1, x2, y2, meta):
    pad_x, pad_y = meta["pad"]
    offset_x, offset_y = meta["offset"]
    scale = meta["scale"]
    return (
        int(round((x1 - pad_x) / scale + offset_x)),
        int(round((y1 - pad_y) / scale + offset_y)),
        int(round((x2 - pad_x) / scale + offset_x)),
        int(round((y2 - pad_y) / scale + offset_y)),
    )
