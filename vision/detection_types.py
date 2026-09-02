import time
from typing import Optional, Tuple


def xyxy_to_xywh(x1, y1, x2, y2):
    return int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))


def bbox_center(bbox):
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def clamp_bbox_xywh(bbox, frame_shape):
    h_img, w_img = frame_shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(0, min(w, w_img - x))
    h = max(0, min(h, h_img - y))
    return x, y, w, h


def make_detection(
    source: str,
    target_type: str,
    bbox,
    confidence: float,
    state: str,
    frame_id: int,
    timestamp: Optional[float] = None,
    error: Optional[Tuple[float, float]] = None,
):
    bbox = tuple(int(round(v)) for v in bbox)
    center = bbox_center(bbox)
    return {
        "source": source,
        "target_type": target_type,
        "bbox": bbox,
        "center": center,
        "confidence": float(confidence),
        "state": state,
        "frame_id": int(frame_id),
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "error": error,
    }
