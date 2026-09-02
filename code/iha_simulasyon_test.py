"""
IHA top-view simulasyon testi.

Webcam/telefon ekrani testi modele haksiz davranabilir. Bu script modelin
yarismaya daha yakin kosullarda ne yaptigini olcer:
  - top-view zemin arka plani
  - kirmizi kare / mavi kare hedef
  - isik, golge, parlama, motion blur, jpeg ve gurultu
  - IoU + sinif dogrulugu raporu

Kullanim:
  python iha_simulasyon_test.py
  python iha_simulasyon_test.py --count 200 --conf 0.25 --imgsz 640
"""

import argparse
import csv
import math
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).parent
ULTRALYTICS_CONFIG_DIR = Path(os.environ.get("ASFLY_ULTRALYTICS_DIR", str(Path(tempfile.gettempdir()) / "asfly_ultralytics")))
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
BG_DIR = SCRIPT_DIR / "backgrounds"
OUT_DIR = SCRIPT_DIR / "simulasyon_sonuclari"

CLASS_NAMES = ["red_square", "blue_square"]
CLASS_COLORS = [(0, 40, 230), (230, 120, 0)]
IMG_SIZE = 640


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def load_backgrounds(bg_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    paths = [p for p in bg_dir.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"Arka plan bulunamadi: {bg_dir}")
    return paths


def random_background(paths):
    img = None
    for _ in range(20):
        img = read_image(random.choice(paths))
        if img is not None:
            break
    if img is None:
        raise RuntimeError("Arka plan okunamadi.")

    h, w = img.shape[:2]
    side = min(h, w)
    if side > 0:
        x = random.randint(0, max(0, w - side))
        y = random.randint(0, max(0, h - side))
        img = img[y:y + side, x:x + side]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return img


def motion_blur(img, length, angle):
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), np.float32)
    c = length // 2
    rad = math.radians(angle)
    dx = math.cos(rad) * c
    dy = math.sin(rad) * c
    cv2.line(
        kernel,
        (int(round(c - dx)), int(round(c - dy))),
        (int(round(c + dx)), int(round(c + dy))),
        1.0,
        1,
    )
    s = kernel.sum()
    if s <= 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def square_points(cx, cy, side, angle_deg):
    half = side / 2.0
    pts = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float32,
    )
    rad = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]],
        dtype=np.float32,
    )
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return np.round(pts).astype(np.int32)


def draw_target(img, cls_id):
    # Kirmizi kare 1m, mavi kare 2m. Mavi pikselde yaklasik 2x buyuk.
    if cls_id == 0:
        side = random.randint(18, 95)
        color = (
            random.randint(0, 35),
            random.randint(0, 45),
            random.randint(185, 255),
        )
    else:
        side = random.randint(32, 175)
        color = (
            random.randint(170, 255),
            random.randint(20, 90),
            random.randint(0, 35),
        )

    margin = side + 20
    cx = random.randint(margin, IMG_SIZE - margin)
    cy = random.randint(margin, IMG_SIZE - margin)
    pts = square_points(cx, cy, side, random.uniform(-35, 35))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, tuple(max(0, c - 55) for c in color), 1)
    x, y, w, h = cv2.boundingRect(pts)
    bbox = (x, y, x + w, y + h)

    return bbox


def apply_conditions(img):
    # Isik/pozlama
    alpha = random.uniform(0.65, 1.35)
    beta = random.randint(-45, 55)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Lokal parlama
    if random.random() < 0.35:
        overlay = img.copy()
        cx = random.randint(0, IMG_SIZE)
        cy = random.randint(0, IMG_SIZE)
        radius = random.randint(45, 160)
        cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), radius / 2)
        img = cv2.addWeighted(img, 0.78, overlay, 0.22, 0)

    # Motion blur ana stres
    if random.random() < 0.70:
        img = motion_blur(img, random.randint(5, 35), random.uniform(0, 180))

    # Defocus/noise/jpeg
    if random.random() < 0.35:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.55:
        noise = np.random.normal(0, random.uniform(4, 25), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.65:
        q = random.randint(35, 88)
        enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="IHA top-view simulasyon testi")
    ap.add_argument("--model", default="best_finetuned.pt")
    ap.add_argument("--bg-dir", default=str(BG_DIR))
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="RTX 4050 icin 0, CPU icin cpu")
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    model = YOLO(str(model_path))

    bg_paths = load_backgrounds(Path(args.bg_dir))
    run_dir = OUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    rows = []
    stats = {
        "total": 0,
        "hit": 0,
        "miss": 0,
        "wrong_class": 0,
        "red_total": 0,
        "red_hit": 0,
        "blue_total": 0,
        "blue_hit": 0,
    }

    for i in range(args.count):
        cls_id = i % 2
        random.shuffle(bg_paths)
        img = random_background(bg_paths)
        gt_bbox = draw_target(img, cls_id)
        img = apply_conditions(img)

        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        preds = []
        for box in result.boxes:
            p_cls = int(box.cls[0])
            score = float(box.conf[0])
            bbox = tuple(float(v) for v in box.xyxy[0].tolist())
            preds.append((p_cls, score, bbox))

        best = None
        for p_cls, score, bbox in preds:
            ov = iou_xyxy(gt_bbox, bbox)
            if best is None or ov > best["iou"]:
                best = {"cls": p_cls, "score": score, "bbox": bbox, "iou": ov}

        ok = best is not None and best["cls"] == cls_id and best["iou"] >= args.iou
        wrong_class = best is not None and best["cls"] != cls_id and best["iou"] >= args.iou

        stats["total"] += 1
        stats["red_total" if cls_id == 0 else "blue_total"] += 1
        if ok:
            stats["hit"] += 1
            stats["red_hit" if cls_id == 0 else "blue_hit"] += 1
        else:
            stats["miss"] += 1
        if wrong_class:
            stats["wrong_class"] += 1

        draw = img.copy()
        cv2.rectangle(draw, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
        cv2.putText(draw, f"GT {CLASS_NAMES[cls_id]}", (gt_bbox[0], max(18, gt_bbox[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if best is not None:
            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            color = CLASS_COLORS[best["cls"]]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f"P {CLASS_NAMES[best['cls']]} {best['score']:.2f} IoU {best['iou']:.2f}",
                        (x1, min(IMG_SIZE - 6, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(draw, "OK" if ok else "FAIL", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0) if ok else (0, 0, 255), 2)

        if i < 40 or not ok:
            cv2.imwrite(str(samples_dir / f"sim_{i:04d}_{'ok' if ok else 'fail'}.jpg"), draw)

        rows.append({
            "idx": i,
            "gt_class": CLASS_NAMES[cls_id],
            "ok": ok,
            "pred_class": CLASS_NAMES[best["cls"]] if best else "",
            "score": f"{best['score']:.4f}" if best else "",
            "iou": f"{best['iou']:.4f}" if best else "0.0000",
        })

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = max(1, stats["total"])
    red_total = max(1, stats["red_total"])
    blue_total = max(1, stats["blue_total"])
    summary = [
        f"Toplam: {stats['total']}",
        f"Hit: {stats['hit']} ({stats['hit'] / total:.1%})",
        f"Miss: {stats['miss']} ({stats['miss'] / total:.1%})",
        f"Wrong class: {stats['wrong_class']}",
        f"Red hit: {stats['red_hit']}/{stats['red_total']} ({stats['red_hit'] / red_total:.1%})",
        f"Blue hit: {stats['blue_hit']}/{stats['blue_total']} ({stats['blue_hit'] / blue_total:.1%})",
        f"CSV: {csv_path}",
        f"Samples: {samples_dir}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
