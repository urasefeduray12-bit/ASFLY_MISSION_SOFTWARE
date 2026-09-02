import argparse
import time
from pathlib import Path

import cv2

import config
from control.payload_logic import compute_drop_ready, release_payload_stub
from control.state_machine import TargetStateMachine, yolo_interval_for_state
from utils.drawing import draw_detections, draw_fused_target, draw_hud, draw_roi
from utils.logger import CsvLogger
from vision.camera import CameraWorker
from vision.fusion import fuse_detections, pick_best_opencv_detection, yolo_result_is_fresh
from vision.opencv_detector import OpenCVDetector
from vision.yolo_detector import YoloAsyncDetector


def resolve_model_path(model_arg):
    cleaned = str(model_arg).strip().strip("\"'")
    while cleaned.endswith(".") and not cleaned.endswith(".pt"):
        cleaned = cleaned[:-1]
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


def parse_args():
    ap = argparse.ArgumentParser(description="Async OpenCV + YOLO fusion target tracker")
    ap.add_argument("--model", default="models_archive/iha_best.pt", help="YOLOv8 .pt model path")
    ap.add_argument("--imgsz", type=int, default=320, help="YOLO input canvas size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--device", default="cpu", help="YOLO device, Raspberry Pi default is cpu")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index fallback")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--headless", action="store_true", help="Do not open cv2 window")
    ap.add_argument("--no-picamera2", action="store_true", help="Skip picamera2 and use cv2.VideoCapture")
    ap.add_argument("--log-csv", default=None, help="Optional CSV log path")
    return ap.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model bulunamadi: {model_path}. --model ile best_finetuned.pt veya models_archive/iha_best.pt verin."
        )

    camera = CameraWorker(
        camera=args.camera,
        width=args.width,
        height=args.height,
        use_picamera2=not args.no_picamera2,
    )
    opencv_detector = OpenCVDetector()
    yolo_detector = YoloAsyncDetector(model_path, imgsz=args.imgsz, conf=args.conf, device=args.device)
    state_machine = TargetStateMachine()
    logger = CsvLogger(args.log_csv)

    frame_id = 0
    fps = 0.0
    fps_counter = 0
    fps_t0 = time.time()
    last_yolo_result = None
    last_yolo_meta = None
    last_yolo_submit_frame = -10**9
    last_drop_print_frame = -10**9

    print("[SAFE MODE] Real payload release disabled. Only WOULD_DROP will be logged.")
    camera.start()
    yolo_detector.start()

    try:
        while True:
            item = camera.read_latest(timeout=1.0)
            if item is None:
                continue
            timestamp, frame = item
            if frame is None:
                raise RuntimeError("Kamera acilamadi. --camera indeksini veya --no-picamera2 ayarini kontrol edin.")

            frame_id += 1
            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0 = now

            vis, opencv_dets = opencv_detector.detect(frame, frame_id)

            new_yolo = yolo_detector.get_latest_result()
            if new_yolo is not None:
                last_yolo_result = new_yolo
                last_yolo_meta = new_yolo.get("meta")

            recent_yolo = yolo_result_is_fresh(last_yolo_result, frame_id)
            yolo_dets = last_yolo_result["detections"] if recent_yolo and last_yolo_result else []
            fused = fuse_detections(opencv_dets, yolo_dets, frame_id)
            cv_target = pick_best_opencv_detection(opencv_dets)

            state = state_machine.update(
                fused,
                has_opencv_target=cv_target is not None,
                has_recent_yolo=recent_yolo,
                drop_ready=False,
            )
            drop_ready = compute_drop_ready(fused, state_machine.lock_counter, state_machine.payload_released)
            if drop_ready:
                state_machine.state = "DROP_READY"
                state = state_machine.state

            if drop_ready and not state_machine.payload_released:
                if frame_id - last_drop_print_frame > 30:
                    release_payload_stub()
                    last_drop_print_frame = frame_id
                state_machine.mark_payload_released()
                state = state_machine.state

            yolo_every = yolo_interval_for_state(state)
            if frame_id - last_yolo_submit_frame >= max(1, yolo_every):
                roi_bbox = cv_target["bbox"] if cv_target is not None else None
                roi_center = cv_target["center"] if cv_target is not None else None
                last_yolo_meta = yolo_detector.submit(
                    frame.copy(),
                    frame_id=frame_id,
                    timestamp=timestamp,
                    roi_center=roi_center,
                    roi_bbox=roi_bbox,
                )
                last_yolo_submit_frame = frame_id

            draw_detections(vis, yolo_dets, (255, 80, 255), "YOLO", 2)
            draw_fused_target(vis, fused)
            draw_roi(vis, last_yolo_meta)

            yolo_age = frame_id - last_yolo_result["frame_id"] if last_yolo_result else 999
            yolo_infer_ms = last_yolo_result.get("infer_ms", 0.0) if last_yolo_result else 0.0
            yolo_error = last_yolo_result.get("error") if last_yolo_result else yolo_detector.load_error
            draw_hud(
                vis,
                state,
                fps,
                yolo_every,
                yolo_age,
                yolo_infer_ms,
                drop_ready,
                state_machine.payload_released,
                yolo_error=str(yolo_error) if yolo_error else None,
            )

            logger.write(
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "state": state,
                    "cv_target": cv_target["target_type"] if cv_target else "",
                    "yolo_target": yolo_dets[0]["target_type"] if yolo_dets else "",
                    "fusion_confidence": fused.get("fusion_confidence", "") if fused else "",
                    "yolo_verified": fused.get("yolo_verified", "") if fused else "",
                    "error_x": fused["error"][0] if fused and fused.get("error") else "",
                    "error_y": fused["error"][1] if fused and fused.get("error") else "",
                    "drop_ready": drop_ready,
                    "fps": fps,
                    "yolo_infer_ms": yolo_infer_ms,
                }
            )

            if not args.headless:
                cv2.imshow("ASFLY Async Fusion SAFE MODE", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        yolo_detector.stop()
        camera.stop()
        logger.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
