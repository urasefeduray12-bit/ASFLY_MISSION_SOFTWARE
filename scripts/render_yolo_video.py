#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO detections into an output video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output annotated video path.")
    parser.add_argument("--model", default="models_archive/iha_best.pt", help="YOLO .pt model path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu or 0.")
    parser.add_argument("--stride", type=int, default=1, help="Run YOLO every N frames; repeated boxes are reused between runs.")
    return parser.parse_args()


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    if label in {"unknown", "square_unknown", "unknown_square"}:
        return "square"
    return label or "square"


def draw_detection(frame, det: dict) -> None:
    x1, y1, x2, y2 = det["xyxy"]
    label = det["label"]
    conf = det["conf"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    model_path = Path(args.model).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    model = YOLO(str(model_path))
    names = {}
    last_detections: list[dict] = []
    frame_id = 0
    stride = max(1, int(args.stride))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            if (frame_id - 1) % stride == 0:
                result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                names = result.names or names
                detections: list[dict] = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "xyxy": (x1, y1, x2, y2),
                            "label": normalize_label(names.get(cls_id, "square")),
                            "conf": conf,
                        }
                    )
                last_detections = detections

            for det in last_detections:
                draw_detection(frame, det)

            cv2.putText(
                frame,
                f"frame={frame_id}/{frame_count or '?'} detections={len(last_detections)} conf={args.conf:.2f} stride={stride}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if frame_id % 50 == 0:
                print(f"processed {frame_id}/{frame_count or '?'} frames")
    finally:
        cap.release()
        writer.release()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
