from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "datasets" / "ds_final_target_square"
DEFAULT_OUT = SCRIPT_DIR / "reports" / "metadata_analysis"
DEFAULT_ARCHIVE = SCRIPT_DIR / "models_archive"
ULTRALYTICS_CONFIG_DIR = Path(
    os.environ.get("ASFLY_ULTRALYTICS_DIR", str(SCRIPT_DIR / ".ultralytics_config"))
)
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


def read_metadata(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("final_split", "") == split]
    return rows


def resolve_model_path(model_arg: str) -> Path:
    if model_arg:
        model_path = Path(model_arg)
        if not model_path.is_absolute():
            model_path = (SCRIPT_DIR / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model bulunamadi: {model_path}")
        return model_path

    candidates = sorted(DEFAULT_ARCHIVE.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "models_archive icinde *_best.pt bulunamadi. --model ile best.pt yolunu ver."
        )
    print(f"Model otomatik secildi: {candidates[0]}")
    return candidates[0]


def read_label(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls, xc, yc, w, h = [float(x) for x in parts]
        except ValueError:
            continue
        if int(cls) != 0:
            continue
        boxes.append([xc, yc, w, h])
    return boxes


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    xc, yc, w, h = box
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    return [x1, y1, x2, y2]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bucket_bbox_size(value: str) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if size <= 0:
        return "none"
    if size < 48:
        return "very_small"
    if size < 80:
        return "small"
    if size < 140:
        return "medium"
    return "large"


def add_group(groups: dict[str, Counter], name: str, key: str, hit: bool, false_positive: bool, has_target: bool) -> None:
    c = groups[f"{name}={key or 'none'}"]
    c["total"] += 1
    c["has_target"] += int(has_target)
    c["hit"] += int(hit)
    c["false_positive"] += int(false_positive)


def summarize_groups(groups: dict[str, Counter]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        c = groups[key]
        has_target = c["has_target"]
        total = c["total"]
        hit = c["hit"]
        fp = c["false_positive"]
        out.append(
            {
                "group": key,
                "total": total,
                "has_target": has_target,
                "hit": hit,
                "recall": round(hit / has_target, 4) if has_target else None,
                "false_positive": fp,
                "false_positive_rate": round(fp / total, 4) if total else None,
            }
        )
    return out


def analyze(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO
    import cv2

    model_path = resolve_model_path(args.model)
    dataset_root = Path(args.dataset).resolve()
    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv bulunamadi: {metadata_path}")

    rows = read_metadata(metadata_path, args.split)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"Analiz icin {args.split} splitinde metadata satiri bulunamadi")

    model = YOLO(str(model_path))
    groups: dict[str, Counter] = defaultdict(Counter)
    details: list[dict[str, Any]] = []
    totals = Counter()

    for index, row in enumerate(rows, start=1):
        image_name = row["final_image_name"]
        image_path = dataset_root / "images" / args.split / image_name
        label_path = dataset_root / "labels" / args.split / f"{Path(image_name).stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in read_label(label_path)]
        has_target = str(row.get("has_target", "")).strip() in {"1", "true", "True"}

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]

        pred_boxes = []
        if result.boxes is not None and len(result.boxes):
            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()

        best_iou = 0.0
        for gt in gt_boxes:
            for pred in pred_boxes:
                best_iou = max(best_iou, iou(gt, pred))

        hit = has_target and best_iou >= args.hit_iou
        false_positive = (not has_target) and bool(pred_boxes)
        missed = has_target and not hit

        totals["total"] += 1
        totals["has_target"] += int(has_target)
        totals["hit"] += int(hit)
        totals["miss"] += int(missed)
        totals["false_positive"] += int(false_positive)

        fields = {
            "source_stage": row.get("source_stage", "none"),
            "target_color": row.get("target_color", "none"),
            "blur_level": row.get("blur_level", "none"),
            "lighting_type": row.get("lighting_type", "none"),
            "shadow_level": row.get("shadow_level", "none"),
            "glare_level": row.get("glare_level", "none"),
            "exposure_level": row.get("exposure_level", "none"),
            "bbox_size_bucket": bucket_bbox_size(row.get("bbox_px_size", "")),
            "has_target": "true" if has_target else "false",
        }
        for name, value in fields.items():
            add_group(groups, name, value, hit, false_positive, has_target)

        details.append(
            {
                "image_name": image_name,
                "has_target": has_target,
                "hit": bool(hit),
                "missed": bool(missed),
                "false_positive": bool(false_positive),
                "pred_count": len(pred_boxes),
                "best_iou": round(best_iou, 4),
                **fields,
            }
        )

        if args.progress and index % args.progress == 0:
            print(f"Analyzed {index}/{len(rows)}")

    out_dir = Path(args.output).resolve() / datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": str(model_path),
        "dataset": str(dataset_root),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "hit_iou": args.hit_iou,
        "totals": dict(totals),
        "overall_recall": round(totals["hit"] / totals["has_target"], 4) if totals["has_target"] else None,
        "false_positive_rate_on_all": round(totals["false_positive"] / totals["total"], 4) if totals["total"] else None,
        "groups": summarize_groups(groups),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()) if details else ["image_name"])
        writer.writeheader()
        writer.writerows(details)

    lines = [
        "ASFLY metadata analysis",
        "=======================",
        f"model: {summary['model']}",
        f"dataset: {summary['dataset']}",
        f"split: {args.split}",
        f"total: {totals['total']}",
        f"has_target: {totals['has_target']}",
        f"hit: {totals['hit']}",
        f"miss: {totals['miss']}",
        f"false_positive: {totals['false_positive']}",
        f"overall_recall: {summary['overall_recall']}",
        f"false_positive_rate_on_all: {summary['false_positive_rate_on_all']}",
        "",
        "Groups:",
    ]
    for item in summary["groups"]:
        lines.append(
            f"- {item['group']}: total={item['total']} has_target={item['has_target']} "
            f"hit={item['hit']} recall={item['recall']} fp={item['false_positive']} "
            f"fp_rate={item['false_positive_rate']}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis ready: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze YOLO target_square model by final metadata")
    ap.add_argument("--model", default="", help="Path to best.pt. Empty means latest models_archive/*_best.pt")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--hit-iou", type=float, default=0.50)
    ap.add_argument("--nms-iou", type=float, default=0.70)
    ap.add_argument("--max-det", type=int, default=10)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="Optional quick test limit")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--progress", type=int, default=250)
    return ap.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
