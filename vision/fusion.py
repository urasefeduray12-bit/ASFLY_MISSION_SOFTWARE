"""Geometry-level fusion helpers.

The functions here match OpenCV detections with YOLO detections using bounding
box IoU. OpenCV keeps color authority; YOLO confirms that the same region is a
square-shaped target.
"""

import config


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pick_best_opencv_detection(detections):
    valid = [d for d in detections if d["source"] in ("opencv", "opencv_kalman")]
    if not valid:
        return None
    return max(valid, key=lambda d: (d["state"] == "DETECTED", d["confidence"]))


def fuse_detections(opencv_detections, yolo_detections, frame_id):
    cv_det = pick_best_opencv_detection(opencv_detections)
    if cv_det is None:
        return None

    best_iou = 0.0
    best_yolo = None
    for yolo_det in yolo_detections or []:
        if yolo_det["target_type"] not in (cv_det["target_type"], "square", "square_unknown"):
            continue
        iou = bbox_iou(cv_det["bbox"], yolo_det["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_yolo = yolo_det

    yolo_conf = best_yolo["confidence"] if best_yolo is not None else 0.0
    yolo_verified = (
        best_yolo is not None
        and best_iou >= config.IOU_VERIFY_THRESH
        and yolo_conf >= config.YOLO_VERIFY_CONF_THRESH
    )
    if yolo_verified:
        fusion_conf = min(1.0, 0.55 * cv_det["confidence"] + 0.45 * yolo_conf + 0.20 * best_iou)
    else:
        fusion_conf = min(0.69, 0.70 * cv_det["confidence"])

    fused = dict(cv_det)
    fused.update(
        {
            "source": "fusion",
            "yolo_verified": bool(yolo_verified),
            "yolo_iou": float(best_iou),
            "fusion_confidence": float(fusion_conf),
            "last_yolo_frame_id": best_yolo["frame_id"] if best_yolo is not None else -1,
            "matched_yolo": best_yolo,
            "age_frames": int(frame_id - cv_det["frame_id"]),
        }
    )
    return fused


def yolo_result_is_fresh(yolo_result, frame_id):
    if yolo_result is None:
        return False
    return frame_id - int(yolo_result.get("frame_id", -10**9)) <= config.MAX_YOLO_RESULT_AGE_FRAMES
