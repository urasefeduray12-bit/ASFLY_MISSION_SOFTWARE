"""
Pi tarzi atlamali YOLO test araci.

Bu dosya ana takip kodunu bozmaz. Amaci:
  - YOLO'yu her frame yerine her N frame'de calistirmak
  - Tracker/HSV varsa YOLO'ya tum goruntu yerine ROI crop vermek
  - "YOLO Input" penceresinde modelin tam olarak gordugu 320x320 goruntuyu gostermek

Kullanim:
  python pi_atlamali_yolo_test.py
  python pi_atlamali_yolo_test.py --model best_finetuned.pt --yolo-every 4
  python pi_atlamali_yolo_test.py --imgsz 256 --yolo-every 5
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).parent
ULTRALYTICS_CONFIG_DIR = Path(os.environ.get("ASFLY_ULTRALYTICS_DIR", str(Path(tempfile.gettempdir()) / "asfly_ultralytics")))
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

CLASS_TR = ["Kirmizi Kare", "Mavi Kare"]
CLASS_COLORS = [(0, 60, 220), (220, 100, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(cam_id: int, width: int, height: int) -> cv2.VideoCapture:
    for backend_name, backend in (
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ):
        cap = cv2.VideoCapture(cam_id, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ok, _ = cap.read()
        if ok:
            print(f"[CAM] Acildi: backend={backend_name}, cam={cam_id}")
            return cap
        cap.release()
    raise RuntimeError(f"Kamera acilamadi: cam={cam_id}")


def make_yolo_input(frame: np.ndarray, imgsz: int, roi_center=None):
    """YOLO'ya verilecek kare canvas'i ve geri-donusum bilgisini hazirla."""
    fh, fw = frame.shape[:2]
    crop_ox = crop_oy = 0

    if roi_center is not None:
        half = imgsz // 2
        rx, ry = roi_center
        x1 = max(0, rx - half)
        x2 = min(fw, rx + half)
        y1 = max(0, ry - half)
        y2 = min(fh, ry + half)

        if x2 - x1 < imgsz:
            x1 = max(0, x2 - imgsz) if x2 == fw else x1
            x2 = min(fw, x1 + imgsz)
        if y2 - y1 < imgsz:
            y1 = max(0, y2 - imgsz) if y2 == fh else y1
            y2 = min(fh, y1 + imgsz)

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        px = (imgsz - cw) // 2
        py = (imgsz - ch) // 2
        canvas[py:py + ch, px:px + cw] = crop
        meta = {
            "mode": "ROI",
            "scale": 1.0,
            "pad": (px, py),
            "offset": (x1, y1),
            "roi_rect": (x1, y1, x2, y2),
        }
        return canvas, meta

    scale = imgsz / max(fw, fh)
    nw, nh = int(fw * scale), int(fh * scale)
    px = (imgsz - nw) // 2
    py = (imgsz - nh) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    meta = {
        "mode": "FULL",
        "scale": scale,
        "pad": (px, py),
        "offset": (0, 0),
        "roi_rect": (0, 0, fw, fh),
    }
    return canvas, meta


def canvas_box_to_frame(x1, y1, x2, y2, meta):
    px, py = meta["pad"]
    ox, oy = meta["offset"]
    scale = meta["scale"]
    if meta["mode"] == "ROI":
        return (
            int(x1 - px + ox),
            int(y1 - py + oy),
            int(x2 - px + ox),
            int(y2 - py + oy),
        )
    return (
        int((x1 - px) / scale),
        int((y1 - py) / scale),
        int((x2 - px) / scale),
        int((y2 - py) / scale),
    )


def robust_color_masks(frame: np.ndarray):
    """HSV + normalize RGB + LAB ile isik degisimine daha dayanikli maskeler."""
    blurred = cv2.GaussianBlur(cv2.medianBlur(frame, 3), (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(blurred)
    _, a_lab, b_lab = cv2.split(lab)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 65]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([168, 80, 65]), np.array([179, 255, 255])),
    )
    blue = cv2.inRange(hsv, np.array([90, 55, 45]), np.array([138, 255, 255]))

    total = b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32) + 1.0
    r_norm = r.astype(np.float32) / total
    b_norm = b.astype(np.float32) / total
    max_gb = np.maximum(g.astype(np.int16), b.astype(np.int16))
    max_rg = np.maximum(r.astype(np.int16), g.astype(np.int16))

    red_dom = (
        ((r.astype(np.int16) - max_gb) > 22) &
        (r > 65) &
        (r_norm > 0.35)
    ) | (
        (a_lab > 146) &
        (r.astype(np.int16) - g.astype(np.int16) > 10) &
        (r > 60)
    )

    blue_dom = (
        ((b.astype(np.int16) - max_rg) > 16) &
        (b > 50) &
        (b_norm > 0.33)
    ) | (
        (b_lab < 120) &
        (b.astype(np.int16) - r.astype(np.int16) > 6) &
        (b > 45)
    )

    red = cv2.bitwise_or(red, red_dom.astype(np.uint8) * 255)
    blue = cv2.bitwise_or(blue, blue_dom.astype(np.uint8) * 255)

    skin = cv2.inRange(hsv, np.array([0, 20, 65]), np.array([24, 175, 255]))
    red = cv2.bitwise_and(red, cv2.bitwise_not(skin))

    kernel9 = np.ones((9, 9), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    for mask in (red, blue):
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel9, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel5, dst=mask)

    return red, blue


def detect_shape_from_mask(mask: np.ndarray, frame_shape, color_name: str):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    min_area = max(900, int(frame_shape[0] * frame_shape[1] * 0.0025))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(1, h)
        fill = area / max(1, w * h)
        if color_name == "red":
            ok_shape = 0.55 <= aspect <= 1.65 and fill >= 0.42
        else:
            ok_shape = 0.65 <= aspect <= 1.55 and fill >= 0.45
        if ok_shape:
            score = fill * max(0.2, 1 - abs(1 - aspect) * 0.30)
            if best is None or score > best["score"]:
                best = {
                    "center": (x + w // 2, y + h // 2),
                    "bbox": (x, y, x + w, y + h),
                    "score": score,
                    "color": color_name,
                }
    return best


def load_pt_model(model_path: Path):
    from ultralytics import YOLO

    print(f"[YOLO] PT model yukleniyor: {model_path}")
    return YOLO(str(model_path))


def run_pt_inference(model, canvas: np.ndarray, conf: float, imgsz: int, device: str):
    result = model.predict(canvas, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    dets = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        dets.append((cls_id, score, x1, y1, x2, y2))
    return dets


def draw_detection(frame, det, meta):
    cls_id, score, x1, y1, x2, y2 = det
    fx1, fy1, fx2, fy2 = canvas_box_to_frame(x1, y1, x2, y2, meta)
    fh, fw = frame.shape[:2]
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(fw - 1, fx2), min(fh - 1, fy2)
    color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (0, 255, 0)
    name = CLASS_TR[cls_id] if cls_id < len(CLASS_TR) else f"class_{cls_id}"
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
    cv2.putText(frame, f"{name} {score:.2f}", (fx1, max(18, fy1 - 6)), FONT, 0.55, color, 2)
    return {
        "center": ((fx1 + fx2) // 2, (fy1 + fy2) // 2),
        "bbox": (fx1, fy1, fx2, fy2),
        "cls": cls_id,
        "score": score,
    }


def draw_yolo_input(canvas, dets, meta, infer_ms, frame_i, yolo_every):
    vis = canvas.copy()
    for cls_id, score, x1, y1, x2, y2 in dets:
        color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (0, 255, 0)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"{cls_id}:{score:.2f}", (int(x1), max(15, int(y1) - 4)),
                    FONT, 0.45, color, 1)
    cv2.putText(vis, f"{meta['mode']}  {infer_ms:.0f}ms", (6, 18), FONT, 0.48, (0, 255, 255), 1)
    cv2.putText(vis, f"YOLO every {yolo_every}f | frame {frame_i}", (6, 38),
                FONT, 0.42, (220, 220, 220), 1)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Pi tarzi atlamali YOLO test")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--model", default="best_finetuned.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0", help="RTX 4050 icin 0, CPU icin cpu")
    ap.add_argument("--yolo-every", type=int, default=4,
                    help="YOLO kac frame'de bir calissin")
    ap.add_argument("--roi-hold", type=int, default=12,
                    help="Son tespiti ROI icin kac frame tasiyalim")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError("Bu test scripti su an PT model ile calisir. ONNX Pi portunu sonra ayri baglariz.")

    model = load_pt_model(model_path)
    cap = open_camera(args.cam, args.width, args.height)

    frame_i = 0
    fps = 0.0
    fps_count = 0
    fps_t0 = time.time()
    last_yolo_dets = []
    last_yolo_canvas = np.full((args.imgsz, args.imgsz, 3), 114, np.uint8)
    last_meta = {"mode": "FULL", "roi_rect": (0, 0, args.width, args.height)}
    last_infer_ms = 0.0
    last_det = None
    last_det_age = 999

    print("[INFO] Q/ESC=cikis, +/-=YOLO araligi, C/V=conf")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        frame_i += 1
        fps_count += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps = fps_count / (now - fps_t0)
            fps_count = 0
            fps_t0 = now

        red_mask, blue_mask = robust_color_masks(frame)
        red_target = detect_shape_from_mask(red_mask, frame.shape, "red")
        blue_target = detect_shape_from_mask(blue_mask, frame.shape, "blue")
        color_targets = [t for t in (red_target, blue_target) if t is not None]
        color_target = max(color_targets, key=lambda t: t["score"]) if color_targets else None

        roi_center = None
        roi_reason = "FULL"
        if last_det is not None and last_det_age <= args.roi_hold:
            roi_center = last_det["center"]
            roi_reason = "LAST_DET"
            last_det_age += 1
        elif color_target is not None:
            roi_center = color_target["center"]
            roi_reason = color_target["color"].upper()

        should_run_yolo = (frame_i % max(1, args.yolo_every)) == 0
        if should_run_yolo:
            canvas, meta = make_yolo_input(frame, args.imgsz, roi_center)
            t0 = time.time()
            last_yolo_dets = run_pt_inference(model, canvas, args.conf, args.imgsz, args.device)
            last_infer_ms = (time.time() - t0) * 1000
            last_yolo_canvas = canvas
            last_meta = meta
            last_meta["reason"] = roi_reason

            if last_yolo_dets:
                best = max(last_yolo_dets, key=lambda d: d[1])
                last_det = draw_detection(frame, best, meta)
                last_det_age = 0
            else:
                last_det = None
                last_det_age = 999
        else:
            if last_det is not None and last_det_age <= args.roi_hold:
                x1, y1, x2, y2 = last_det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
                cv2.putText(frame, f"HOLD {last_det_age}/{args.roi_hold}",
                            (x1, max(18, y1 - 6)), FONT, 0.45, (200, 200, 0), 1)

        if red_target is not None:
            x1, y1, x2, y2 = red_target["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 1)
            cv2.putText(frame, f"CV red {red_target['score']:.2f}",
                        (x1, min(frame.shape[0] - 6, y2 + 16)), FONT, 0.45, (0, 80, 255), 1)
        if blue_target is not None:
            x1, y1, x2, y2 = blue_target["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(frame, f"CV blue {blue_target['score']:.2f}",
                        (x1, min(frame.shape[0] - 6, y2 + 16)), FONT, 0.45, (255, 180, 0), 1)

        rx1, ry1, rx2, ry2 = last_meta.get("roi_rect", (0, 0, frame.shape[1], frame.shape[0]))
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

        cv2.putText(frame, f"FPS:{fps:.1f}  YOLO every:{args.yolo_every}f  Conf:{args.conf:.2f}",
                    (10, 24), FONT, 0.55, (0, 255, 255), 1)
        cv2.putText(frame, f"ROI:{last_meta.get('mode', 'FULL')} reason:{last_meta.get('reason', '-')}"
                    f"  infer:{last_infer_ms:.0f}ms",
                    (10, 48), FONT, 0.50, (230, 230, 230), 1)

        yolo_vis = draw_yolo_input(last_yolo_canvas, last_yolo_dets, last_meta,
                                   last_infer_ms, frame_i, args.yolo_every)

        cv2.imshow("Pi Atlamali Test", frame)
        cv2.imshow("YOLO Input - modelin gordugu", yolo_vis)
        cv2.imshow("Red CV Mask", cv2.resize(red_mask, (args.width, args.height)))
        cv2.imshow("Blue CV Mask", cv2.resize(blue_mask, (args.width, args.height)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            args.yolo_every = max(1, args.yolo_every - 1)
            print(f"[INFO] YOLO every -> {args.yolo_every}")
        elif key == ord("-"):
            args.yolo_every = min(12, args.yolo_every + 1)
            print(f"[INFO] YOLO every -> {args.yolo_every}")
        elif key == ord("c"):
            args.conf = max(0.05, args.conf - 0.05)
            print(f"[INFO] conf -> {args.conf:.2f}")
        elif key == ord("v"):
            args.conf = min(0.95, args.conf + 0.05)
            print(f"[INFO] conf -> {args.conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
