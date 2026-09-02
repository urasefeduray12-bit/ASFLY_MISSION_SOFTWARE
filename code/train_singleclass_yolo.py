from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "datasets" / "ds_final_target_square" / "data.yaml"
DEFAULT_MODEL = SCRIPT_DIR / "yolov8n.pt"
DEFAULT_PROJECT = SCRIPT_DIR / "runs_singleclass"
ARCHIVE_DIR = SCRIPT_DIR / "models_archive"
ULTRALYTICS_CONFIG_DIR = Path(
    os.environ.get("ASFLY_ULTRALYTICS_DIR", str(SCRIPT_DIR / ".ultralytics_config"))
)
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML bulunamadi. Ultralytics ortaminda normalde yuklu olmali.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"data.yaml beklenen formatta degil: {path}")
    return data


def resolve_dataset_root(data_yaml: Path, yaml_data: dict[str, Any]) -> Path:
    raw_path = yaml_data.get("path", "")
    if raw_path:
        root = Path(str(raw_path))
        if not root.is_absolute():
            candidate_from_cwd = (Path.cwd() / root).resolve()
            candidate_from_script = (SCRIPT_DIR / root).resolve()
            candidate_from_yaml_parent = (data_yaml.parent / root).resolve()
            if candidate_from_script.exists():
                return candidate_from_script
            if candidate_from_cwd.exists():
                return candidate_from_cwd
            if candidate_from_yaml_parent.exists():
                return candidate_from_yaml_parent
            return candidate_from_script
        return root
    return data_yaml.parent


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def count_files(path: Path, exts: set[str] | None = None) -> int:
    if not path.is_dir():
        return 0
    if exts is None:
        return sum(1 for p in path.iterdir() if p.is_file())
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_label_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_counts(dataset_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        counts[split] = {
            "images": count_files(image_dir, IMAGE_EXTS),
            "labels": count_files(label_dir, {".txt"}),
        }
    return counts


def summarize_sanity_report(dataset_root: Path) -> list[str]:
    report = dataset_root / "sanity_report.txt"
    if not report.exists():
        return ["sanity_report.txt bulunamadi"]
    lines = report.read_text(encoding="utf-8", errors="replace").splitlines()
    keep: list[str] = []
    prefixes = (
        "Total images:",
        "Split counts:",
        "Stage counts:",
        "Target color counts:",
        "Has target counts:",
        "Warnings:",
        "Errors:",
    )
    active = False
    for line in lines:
        if any(line.startswith(prefix) for prefix in prefixes):
            active = True
            keep.append(line)
            continue
        if active and line.startswith("  "):
            keep.append(line)
            continue
        active = False
    return keep[:80]


def validate_labels(dataset_root: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    problems: list[str] = []
    class_counts: dict[str, int] = {}
    empty_label_count = 0
    nonempty_label_count = 0
    missing_label_count = 0
    bbox_out_of_range = 0

    metadata_has_target: dict[str, bool] = {}
    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                final_name = row.get("final_image_name", "").strip()
                split = row.get("final_split", "").strip()
                if final_name and split:
                    metadata_has_target[f"{split}/{final_name}"] = str(row.get("has_target", "")).strip() in {
                        "1",
                        "true",
                        "True",
                    }

    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        for image_path in sorted(image_dir.iterdir() if image_dir.exists() else []):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                problems.append(f"Label eksik: {label_path}")
                missing_label_count += 1
                continue
            lines = read_label_lines(label_path)
            if lines:
                nonempty_label_count += 1
            else:
                empty_label_count += 1

            key = f"{split}/{image_path.name}"
            if key in metadata_has_target:
                has_target = metadata_has_target[key]
                if has_target and not lines:
                    problems.append(f"Metadata target diyor ama label bos: {key}")
                if not has_target and lines:
                    problems.append(f"Metadata target yok diyor ama label dolu: {key}")

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"Label formati hatali: {label_path}: {line}")
                    continue
                class_id = parts[0]
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                if class_id != "0":
                    problems.append(f"Class id 0 degil: {label_path}: {line}")
                try:
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    problems.append(f"BBox sayisal degil: {label_path}: {line}")
                    continue
                if any(v < 0.0 or v > 1.0 for v in vals):
                    bbox_out_of_range += 1
                    problems.append(f"BBox 0-1 araligi disinda: {label_path}: {line}")

    return {
        "class_counts": class_counts,
        "empty_label_count": empty_label_count,
        "nonempty_label_count": nonempty_label_count,
        "missing_label_count": missing_label_count,
        "bbox_out_of_range": bbox_out_of_range,
        "problems": problems,
    }


def print_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"device_summary": "unknown"}
    try:
        import torch

        cuda = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(cuda)
        if cuda:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["device_summary"] = f"CUDA: {info['cuda_device_name']}"
        else:
            info["device_summary"] = "CPU"
    except Exception as exc:
        info["torch_error"] = str(exc)

    print("\nHardware:")
    print(f"  {info.get('device_summary')}")
    if info.get("torch_version"):
        print(f"  torch: {info['torch_version']}")
    return info


def preflight_checks(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml bulunamadi: {data_yaml}")

    yaml_data = read_yaml(data_yaml)
    names = normalize_names(yaml_data.get("names"))
    nc = int(yaml_data.get("nc", -1))
    if nc != 1:
        raise ValueError(f"data.yaml nc=1 olmali, bulundu: {nc}")
    if names.get(0) != "target_square":
        raise ValueError(f"names[0] target_square olmali, bulundu: {names.get(0)!r}")

    dataset_root = resolve_dataset_root(data_yaml, yaml_data)
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("Eksik dataset klasorleri:\n" + "\n".join(missing_dirs))

    metadata_path = dataset_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv bulunamadi: {metadata_path}")

    label_check = validate_labels(dataset_root, metadata_path)
    if label_check["problems"]:
        preview = "\n".join(label_check["problems"][:30])
        raise ValueError(f"Dataset label kontrolu basarisiz:\n{preview}")

    counts = dataset_counts(dataset_root)
    hardware = print_hardware()

    print("\nDataset:")
    print(f"  data.yaml: {data_yaml}")
    print(f"  root     : {dataset_root}")
    print(f"  nc/names : {nc}, {names}")
    for split, split_counts in counts.items():
        print(f"  {split:5s}: images={split_counts['images']} labels={split_counts['labels']}")
    print(f"  class ids: {label_check['class_counts']}")
    print(f"  empty labels: {label_check['empty_label_count']}")
    print(f"  nonempty labels: {label_check['nonempty_label_count']}")

    print("\nSanity report summary:")
    for line in summarize_sanity_report(dataset_root):
        print(f"  {line}")

    return {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "yaml": yaml_data,
        "counts": counts,
        "label_check": label_check,
        "hardware": hardware,
    }


def timestamped_name(base_name: str, no_timestamp: bool) -> str:
    clean = base_name.strip() or "target_square_yolov8n"
    if no_timestamp:
        return clean
    return f"{clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        out.update({str(k): float(v) if isinstance(v, (int, float)) else v for k, v in results_dict.items()})
    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("mp", "mr", "map50", "map", "map75"):
            value = getattr(box, attr, None)
            if value is not None:
                try:
                    out[f"box_{attr}"] = float(value)
                except TypeError:
                    out[f"box_{attr}"] = str(value)
    return out


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_absolute_data_yaml(original_data_yaml: Path, dataset_root: Path, run_name: str) -> Path:
    import yaml

    data = read_yaml(original_data_yaml)
    data["path"] = str(dataset_root)
    out_dir = SCRIPT_DIR / "reports" / "runtime_data_yamls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_data.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_path


def write_predict_source_list(dataset_root: Path, split: str, run_name: str, seed: int, count: int = 50) -> Path:
    image_dir = dataset_root / "images" / split
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(seed)
    rng.shuffle(images)
    selected = images[: min(count, len(images))]
    list_path = SCRIPT_DIR / "reports" / f"{run_name}_predict_source.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(str(p) for p in selected), encoding="utf-8")
    return list_path


def archive_model(run_name: str, save_dir: Path, args: argparse.Namespace, context: dict[str, Any], val_metrics: Any, test_metrics: Any) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt bulunamadi: {best_pt}")

    archive_base = ARCHIVE_DIR / run_name
    shutil.copy2(best_pt, ARCHIVE_DIR / f"{run_name}_best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, ARCHIVE_DIR / f"{run_name}_last.pt")

    metrics_payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "save_dir": str(save_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt) if last_pt.exists() else "",
        "val_metrics": metrics_to_dict(val_metrics),
        "test_metrics": metrics_to_dict(test_metrics),
        "dataset_counts": context.get("counts", {}),
        "hardware": context.get("hardware", {}),
    }
    save_json(Path(f"{archive_base}_metrics.json"), metrics_payload)

    train_args = vars(args).copy()
    train_args["resolved_run_name"] = run_name
    train_args["created_at"] = datetime.now().isoformat(timespec="seconds")
    train_args["dataset_context"] = {
        "data_yaml": context.get("data_yaml"),
        "dataset_root": context.get("dataset_root"),
        "counts": context.get("counts", {}),
    }
    save_json(Path(f"{archive_base}_train_args.json"), train_args)


def run_training(args: argparse.Namespace, context: dict[str, Any]) -> None:
    from ultralytics import YOLO

    original_data_yaml = Path(args.data).resolve()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (SCRIPT_DIR / model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Baslangic modeli bulunamadi: {model_path}")

    run_name = timestamped_name(args.name, args.no_timestamp)
    project = Path(args.project)
    if not project.is_absolute():
        project = (SCRIPT_DIR / project).resolve()
    dataset_root = Path(context["dataset_root"])
    data_yaml = write_absolute_data_yaml(original_data_yaml, dataset_root, run_name)

    print("\nTraining:")
    print(f"  model  : {model_path}")
    print(f"  data   : {data_yaml}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  epochs : {args.epochs}")
    print(f"  batch  : {args.batch}")
    print(f"  device : {args.device}")
    print(f"  workers: {args.workers}")
    print(f"  run    : {project / run_name}")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = (SCRIPT_DIR / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume modeli bulunamadi: {resume_path}")
        model = YOLO(str(resume_path))
        results = model.train(resume=True)
    else:
        model = YOLO(str(model_path))
        results = model.train(
            data=str(data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            device=args.device,
            project=str(project),
            name=run_name,
            exist_ok=False,
            seed=args.seed,
            deterministic=True,
            cache=args.cache,
            plots=True,
            val=True,
            mixup=0.0,
            copy_paste=0.0,
            mosaic=args.mosaic,
            close_mosaic=args.close_mosaic,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            flipud=args.flipud,
        )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Egitim bitti ama best.pt yok: {best_pt}")

    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        project=str(project),
        name=f"{run_name}_val",
        plots=True,
        exist_ok=False,
        workers=args.workers,
        device=args.device,
    )
    test_metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        project=str(project),
        name=f"{run_name}_test",
        plots=True,
        exist_ok=False,
        workers=args.workers,
        device=args.device,
    )

    preview_source = write_predict_source_list(dataset_root, "test", run_name, args.seed, count=50)
    predict_name = f"{run_name}_predict50"
    best_model.predict(
        source=str(preview_source),
        imgsz=args.imgsz,
        conf=args.predict_conf,
        max_det=10,
        project=str(project),
        name=predict_name,
        save=True,
        stream=False,
        device=args.device,
        verbose=False,
    )

    archive_model(run_name, save_dir, args, context, val_metrics, test_metrics)
    print(f"\nDone. Run: {save_dir}")
    print(f"Archive: {ARCHIVE_DIR}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train single-class YOLOv8n target_square model")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Final dataset data.yaml path")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Start model, usually yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8, help="Batch size, or -1 for Ultralytics auto batch")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--name", default="target_square_yolov8n_320_baseline")
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", default="", help="Reserved for manual resume path; not used by default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-check", action="store_true", help="Only run dataset/GPU checks, do not train")
    ap.add_argument("--no-timestamp", action="store_true", help="Use exact run name")
    ap.add_argument("--cache", action="store_true", help="Enable Ultralytics cache")
    ap.add_argument("--mosaic", type=float, default=0.4)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--hsv-h", type=float, default=0.005)
    ap.add_argument("--hsv-s", type=float, default=0.25)
    ap.add_argument("--hsv-v", type=float, default=0.20)
    ap.add_argument("--degrees", type=float, default=3.0)
    ap.add_argument("--translate", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.20)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--predict-conf", type=float, default=0.25)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    context = preflight_checks(Path(args.data).resolve())
    if args.dry_check:
        print("\nDry check tamam. Egitim baslatilmadi.")
        return
    run_training(args, context)


if __name__ == "__main__":
    main()
