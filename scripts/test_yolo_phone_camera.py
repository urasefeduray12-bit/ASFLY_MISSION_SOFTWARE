#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the single-class square YOLO model on a phone/IP camera stream.")
    parser.add_argument(
        "--source",
        required=True,
        help="Camera source. Use an IP camera URL such as http://PHONE_IP:8080/video, or 0 for a local webcam.",
    )
    parser.add_argument("--model", default="models_archive/iha_best.pt", help="Path to YOLO .pt model.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="cpu", help="YOLO device, for example cpu or 0.")
    parser.add_argument("--save", default="", help="Optional output video path.")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a preview window.")
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    if label in {"unknown", "square_unknown", "unknown_square"}:
        return "square"
    return label or "square"


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(resolve_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.source}")

    writer = None
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    frame_id = 0
    last_time = time.perf_counter()
    fps_smoothed = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; check phone stream/Wi-Fi connection.")
                time.sleep(0.2)
                continue

            frame_id += 1
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            detections = 0
            names = result.names or {}

            for box in result.boxes:
                detections += 1
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = normalize_label(names.get(cls_id, "square"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.perf_counter()
            instant_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            fps_smoothed = instant_fps if fps_smoothed == 0.0 else 0.9 * fps_smoothed + 0.1 * instant_fps
            cv2.putText(
                frame,
                f"frame={frame_id} detections={detections} conf={args.conf:.2f} fps={fps_smoothed:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("YOLO square phone camera test", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
