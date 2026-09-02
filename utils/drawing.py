import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_OPENCV = (0, 180, 255)
COLOR_YOLO = (255, 80, 255)
COLOR_FUSION = (0, 220, 0)
COLOR_ROI = (0, 255, 255)
COLOR_TEXT = (235, 235, 235)
COLOR_WARN = (0, 0, 255)


def draw_detections(frame, detections, color, prefix, thickness=2):
    for det in detections or []:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(frame, det["center"], 4, color, -1)
        cv2.putText(
            frame,
            f"{prefix} {det['target_type']} {det['confidence']:.2f}",
            (x, max(16, y - 6)),
            FONT,
            0.45,
            color,
            1,
        )


def draw_fused_target(frame, fused):
    if fused is None:
        return
    x, y, w, h = fused["bbox"]
    color = COLOR_FUSION if fused.get("yolo_verified") else COLOR_WARN
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.circle(frame, fused["center"], 5, color, -1)
    err = fused.get("error")
    if err is not None:
        cx, cy = fused["center"]
        cv2.arrowedLine(frame, (cx, cy), (cx - int(err[0]), cy - int(err[1])), color, 2, tipLength=0.25)
    cv2.putText(
        frame,
        f"FUSION {fused.get('fusion_confidence', 0):.2f} IoU {fused.get('yolo_iou', 0):.2f}",
        (x, min(frame.shape[0] - 8, y + h + 18)),
        FONT,
        0.50,
        color,
        1,
    )


def draw_roi(frame, meta):
    if not meta:
        return
    x1, y1, x2, y2 = meta.get("roi_rect", (0, 0, frame.shape[1], frame.shape[0]))
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_ROI, 1)
    cv2.putText(frame, meta.get("mode", "-"), (int(x1) + 4, max(16, int(y1) + 16)), FONT, 0.45, COLOR_ROI, 1)


def draw_hud(
    frame,
    state,
    fps,
    yolo_every,
    yolo_age,
    yolo_infer_ms,
    drop_ready,
    payload_released,
    yolo_error=None,
):
    lines = [
        f"STATE: {state}",
        f"FPS: {fps:.1f}  YOLO every: {yolo_every}f  age: {yolo_age}",
        f"YOLO infer: {yolo_infer_ms:.0f} ms",
    ]
    if yolo_error:
        lines.append(f"YOLO error: {yolo_error[:72]}")
    if drop_ready:
        lines.append("DROP_READY / WOULD_DROP")
    if payload_released:
        lines.append("DROP_EXECUTED (SIMULATED)")

    y = 24
    for line in lines:
        color = COLOR_WARN if "DROP" in line or "error" in line else COLOR_TEXT
        cv2.putText(frame, line, (10, y), FONT, 0.58, color, 2 if "STATE" in line else 1)
        y += 24
