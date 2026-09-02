"""
Build the final single-class YOLO dataset from approved curriculum stages.

This script does not modify source stage datasets. It only reads approved stage
folders, samples/copies images and labels into a final YOLO layout, normalizes
labels to class 0 in the final copy, and writes metadata plus sanity reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS_ROOT = SCRIPT_DIR / "datasets"

STAGE_SOURCES = {
    "stage1": "ds_stage1_shape_basic",
    "stage2": "ds_stage2_field_clean_v2_preview",
    "stage3": "ds_stage3_field_blur_clean_motion_v2_preview",
    "stage4": "ds_stage4_field_sun_shadow_controlled_v2_preview",
}

STAGE_RATIOS = {
    "stage1": 0.10,
    "stage2": 0.40,
    "stage3": 0.25,
    "stage4": 0.25,
}

MODE_TOTALS = {
    "debug": None,
    "mini_final": 24000,
    "full_final": 42000,
}

FULL_FINAL_STAGE_COUNTS = {
    "stage1": 4000,
    "stage2": 16000,
    "stage3": 10000,
    "stage4": 12000,
}

FINAL_METADATA_FIELDS = [
    "sample_id",
    "source_stage",
    "original_image_name",
    "original_split",
    "final_image_name",
    "final_split",
    "has_target",
    "target_color",
    "target_type",
    "background_type",
    "negative_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_px_size",
    "rotation_deg",
    "perspective_level",
    "blur_type",
    "blur_level",
    "motion_blur_length",
    "motion_blur_angle",
    "lighting_type",
    "lighting_level",
    "shadow_level",
    "glare_level",
    "exposure_level",
    "color_shift_level",
    "notes",
    "image_hash",
    "label_hash",
]

EXCLUDED_NAME_PARTS = (
    "preview",
    "debug",
    "annotated",
    "vis",
    "visualized",
    "grid",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tif", ".tiff"}


@dataclass
class Sample:
    source_stage: str
    stage_path: Path
    image_path: Path
    label_path: Path
    original_image_name: str
    original_split: str
    metadata: dict[str, str]
    image_hash: str = ""
    label_hash: str = ""
    final_split: str = ""
    final_image_name: str = ""
    sample_id: str = ""
    normalized_label_lines: list[str] = field(default_factory=list)
    converted_label_count: int = 0
    label_warnings: list[str] = field(default_factory=list)


def truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_label_lines(lines: list[str]) -> str:
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_empty(value: object, default: str = "none") -> str:
    text = "" if value is None else str(value).strip()
    return text if text else default


def load_stage_metadata(stage_path: Path) -> list[dict[str, str]]:
    metadata_path = stage_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv bulunamadi: {metadata_path}")
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_excluded_training_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in EXCLUDED_NAME_PARTS)


def collect_stage_samples(stage_name: str, stage_path: Path) -> tuple[list[Sample], list[str]]:
    warnings: list[str] = []
    rows = load_stage_metadata(stage_path)
    samples: list[Sample] = []

    for row in rows:
        image_name = row.get("image_name", "").strip()
        split = row.get("split", "").strip() or "unknown"
        if not image_name:
            warnings.append(f"{stage_name}: image_name bos metadata satiri atlandi")
            continue
        if is_excluded_training_name(image_name):
            warnings.append(f"{stage_name}: preview/debug isimli dosya dislandi: {image_name}")
            continue
        if Path(image_name).suffix.lower() not in IMAGE_EXTS:
            warnings.append(f"{stage_name}: desteklenmeyen image uzantisi dislandi: {image_name}")
            continue

        image_path = stage_path / "images" / split / image_name
        label_path = stage_path / "labels" / split / f"{Path(image_name).stem}.txt"
        if not image_path.exists():
            warnings.append(f"{stage_name}: image bulunamadi: {image_path}")
            continue

        samples.append(
            Sample(
                source_stage=stage_name,
                stage_path=stage_path,
                image_path=image_path,
                label_path=label_path,
                original_image_name=image_name,
                original_split=split,
                metadata=row,
            )
        )

    return samples, warnings


def collect_all_stage_samples(datasets_root: Path) -> tuple[dict[str, list[Sample]], list[str]]:
    by_stage: dict[str, list[Sample]] = {}
    warnings: list[str] = []
    for stage_name, folder_name in STAGE_SOURCES.items():
        stage_path = datasets_root / folder_name
        if not stage_path.exists():
            warnings.append(f"{stage_name}: kaynak klasor bulunamadi: {stage_path}")
            by_stage[stage_name] = []
            continue
        samples, stage_warnings = collect_stage_samples(stage_name, stage_path)
        by_stage[stage_name] = samples
        warnings.extend(stage_warnings)
    return by_stage, warnings


def normalize_label_to_single_class(sample: Sample) -> tuple[list[str], int, list[str]]:
    warnings: list[str] = []
    has_target = truthy(sample.metadata.get("has_target", "0"))

    if not sample.label_path.exists():
        if has_target:
            warnings.append(f"label eksik ama has_target=true: {sample.label_path}")
        return [], 0, warnings

    text = sample.label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], 0, warnings

    out_lines: list[str] = []
    converted = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"label formati bozuk: {sample.label_path}:{line_no}")
            continue
        cls = parts[0]
        if cls != "0":
            converted += 1
            cls = "0"
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            warnings.append(f"label sayisal degil: {sample.label_path}:{line_no}")
            continue
        if any(v < 0.0 or v > 1.0 for v in vals):
            warnings.append(f"bbox 0-1 araliginda degil: {sample.label_path}:{line_no}")
            continue
        out_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in vals))

    return out_lines, converted, warnings


def filter_valid_training_images(samples: list[Sample]) -> tuple[list[Sample], list[str], Counter]:
    valid: list[Sample] = []
    warnings: list[str] = []
    stats: Counter = Counter()

    for sample in samples:
        lines, converted, label_warnings = normalize_label_to_single_class(sample)
        sample.normalized_label_lines = lines
        sample.converted_label_count = converted
        sample.label_warnings = label_warnings
        sample.image_hash = file_sha256(sample.image_path)
        sample.label_hash = hash_label_lines(lines)
        has_target = truthy(sample.metadata.get("has_target", "0"))

        if label_warnings:
            warnings.extend([f"{sample.source_stage}/{sample.original_image_name}: {w}" for w in label_warnings])
        if has_target and not lines:
            warnings.append(f"{sample.source_stage}/{sample.original_image_name}: has_target=true ama label bos/gecersiz")
            stats["target_with_empty_label"] += 1
            continue
        if not has_target and lines:
            warnings.append(f"{sample.source_stage}/{sample.original_image_name}: has_target=false ama label dolu")
            stats["empty_with_label"] += 1
            continue
        if converted:
            stats["converted_label_lines"] += converted
        valid.append(sample)

    return valid, warnings, stats


def mode_target_counts(mode: str, max_per_stage: int) -> dict[str, int]:
    if mode == "debug":
        return {stage: max_per_stage for stage in STAGE_SOURCES}
    if mode == "full_final":
        return dict(FULL_FINAL_STAGE_COUNTS)
    total = MODE_TOTALS[mode]
    assert total is not None
    counts = {stage: int(round(total * ratio)) for stage, ratio in STAGE_RATIOS.items()}
    diff = total - sum(counts.values())
    if diff:
        counts["stage2"] += diff
    return counts


def sample_by_stage_ratio(
    samples_by_stage: dict[str, list[Sample]],
    target_counts: dict[str, int],
    rng: random.Random,
) -> tuple[list[Sample], list[str]]:
    selected: list[Sample] = []
    warnings: list[str] = []
    for stage_name in STAGE_SOURCES:
        pool = list(samples_by_stage.get(stage_name, []))
        rng.shuffle(pool)
        wanted = target_counts.get(stage_name, 0)
        if len(pool) < wanted:
            warnings.append(f"{stage_name}: hedef {wanted}, mevcut {len(pool)}; mevcut kadar alinacak")
        selected.extend(pool[: min(wanted, len(pool))])
    rng.shuffle(selected)
    return selected, warnings


def stratify_key(sample: Sample) -> tuple[str, str, str, str, str]:
    m = sample.metadata
    return (
        sample.source_stage,
        safe_empty(m.get("target_color")),
        safe_empty(m.get("target_type")),
        safe_empty(m.get("blur_level")),
        safe_empty(m.get("lighting_type")),
    )


def split_for_position(pos: int, total: int) -> str:
    ratio = pos / max(1, total)
    if ratio < 0.80:
        return "train"
    if ratio < 0.90:
        return "val"
    return "test"


def create_final_split(samples: list[Sample], rng: random.Random) -> None:
    groups: dict[tuple[str, str, str, str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[stratify_key(sample)].append(sample)

    # Split each semantic bucket separately so empty/negative/blur/light samples
    # remain represented in val/test instead of being swallowed by train.
    for group in groups.values():
        rng.shuffle(group)
        total = len(group)
        train_count = int(round(total * 0.80))
        val_count = int(round(total * 0.10))
        for pos, sample in enumerate(group):
            if pos < train_count:
                sample.final_split = "train"
            elif pos < train_count + val_count:
                sample.final_split = "val"
            else:
                sample.final_split = "test"


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Final dataset zaten var: {out_dir} --overwrite ile yeniden olustur")
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def assign_final_names(samples: list[Sample]) -> None:
    counters: Counter = Counter()
    for sample in samples:
        counters[sample.source_stage] += 1
        stem = f"{sample.source_stage}_{counters[sample.source_stage]:06d}"
        ext = sample.image_path.suffix.lower()
        sample.sample_id = stem
        sample.final_image_name = f"{stem}{ext}"


def copy_image_and_label(sample: Sample, out_dir: Path) -> None:
    image_dst = out_dir / "images" / sample.final_split / sample.final_image_name
    label_dst = out_dir / "labels" / sample.final_split / f"{Path(sample.final_image_name).stem}.txt"
    shutil.copy2(sample.image_path, image_dst)
    label_text = "\n".join(sample.normalized_label_lines)
    if label_text:
        label_text += "\n"
    label_dst.write_text(label_text, encoding="utf-8")


def metadata_value(row: dict[str, str], key: str, default: str = "none") -> str:
    return safe_empty(row.get(key), default)


def merge_metadata(samples: list[Sample]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in samples:
        m = sample.metadata
        rows.append({
            "sample_id": sample.sample_id,
            "source_stage": sample.source_stage,
            "original_image_name": sample.original_image_name,
            "original_split": sample.original_split,
            "final_image_name": sample.final_image_name,
            "final_split": sample.final_split,
            "has_target": str(int(truthy(m.get("has_target", "0")))),
            "target_color": metadata_value(m, "target_color"),
            "target_type": metadata_value(m, "target_type"),
            "background_type": metadata_value(m, "background_type"),
            "negative_type": metadata_value(m, "negative_type"),
            "bbox_x": metadata_value(m, "bbox_x", "0"),
            "bbox_y": metadata_value(m, "bbox_y", "0"),
            "bbox_w": metadata_value(m, "bbox_w", "0"),
            "bbox_h": metadata_value(m, "bbox_h", "0"),
            "bbox_px_size": metadata_value(m, "bbox_px_size", "0"),
            "rotation_deg": metadata_value(m, "rotation_deg", "0"),
            "perspective_level": metadata_value(m, "perspective_level", "0"),
            "blur_type": metadata_value(m, "blur_type"),
            "blur_level": metadata_value(m, "blur_level"),
            "motion_blur_length": metadata_value(m, "motion_blur_length", "0"),
            "motion_blur_angle": metadata_value(m, "motion_blur_angle", "0"),
            "lighting_type": metadata_value(m, "lighting_type"),
            "lighting_level": metadata_value(m, "lighting_level"),
            "shadow_level": metadata_value(m, "shadow_level"),
            "glare_level": metadata_value(m, "glare_level"),
            "exposure_level": metadata_value(m, "exposure_level"),
            "color_shift_level": metadata_value(m, "color_shift_level"),
            "notes": metadata_value(m, "notes", ""),
            "image_hash": sample.image_hash,
            "label_hash": sample.label_hash,
        })
    return rows


def write_metadata(out_dir: Path, rows: list[dict[str, str]]) -> None:
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(out_dir: Path) -> None:
    try:
        path_value = out_dir.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        path_value = str(out_dir.resolve())
    (out_dir / "data.yaml").write_text(
        f"""path: {path_value}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: target_square
""",
        encoding="utf-8",
    )


def write_readme(out_dir: Path, mode: str, target_counts: dict[str, int]) -> None:
    lines = [
        "# ASFLY Final Target Square Dataset",
        "",
        "YOLO tek siniflidir: `0: target_square`.",
        "",
        "Kirmizi 1x1 m kare ve mavi 2x2 m kare ayni YOLO sinifi olarak egitilir.",
        "Renk karari egitimde YOLO'ya verilmez; gercek sistemde YOLO bbox crop'u uzerinden OpenCV HSV/LAB analiziyle yapilir.",
        "",
        "## Label format",
        "",
        "```txt",
        "0 x_center y_center width height",
        "```",
        "",
        "Bos/background/negative goruntuler icin label dosyasi vardir ama bostur.",
        "",
        "## Kaynak stage hedef oranlari",
        "",
        "- Stage 1 shape/basic: %10",
        "- Stage 2 field clean: %40",
        "- Stage 3 blur: %25",
        "- Stage 4 sun/shadow/glare: %25",
        "",
        f"Uretim modu: `{mode}`",
        f"Hedef stage sayilari: `{dict(target_counts)}`",
        "",
        "## Split",
        "",
        "Final split hedefi train/val/test = 80/10/10. Split final dataset olusturulurken yeniden yapilir.",
        "",
        "## Metadata",
        "",
        "metadata.csv icinde source stage, original dosya adi, final dosya adi, renk bilgisi, bbox piksel bilgisi, blur ve lighting bilgileri tutulur.",
        "",
        "## Egitim",
        "",
        "Ultralytics YOLO egitiminde `data.yaml` dosyasini kullan.",
    ]
    (out_dir / "README_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_yolo_bbox(label_line: str) -> tuple[float, float, float, float] | None:
    parts = label_line.split()
    if len(parts) != 5:
        return None
    try:
        return tuple(float(v) for v in parts[1:5])  # type: ignore[return-value]
    except ValueError:
        return None


def create_final_preview(out_dir: Path, rows: list[dict[str, str]], max_items: int = 30) -> None:
    if not rows:
        return
    rng = random.Random(123)
    by_stage: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stage[row["source_stage"]].append(row)
    preview_rows: list[dict[str, str]] = []
    per_stage = max(1, math.ceil(max_items / max(1, len(by_stage))))
    for stage in sorted(by_stage):
        pool = list(by_stage[stage])
        rng.shuffle(pool)
        preview_rows.extend(pool[:per_stage])
    preview_rows = preview_rows[:max_items]

    thumbs: list[np.ndarray] = []
    for row in preview_rows:
        split = row["final_split"]
        img_path = out_dir / "images" / split / row["final_image_name"]
        label_path = out_dir / "labels" / split / f"{Path(row['final_image_name']).stem}.txt"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vis = img.copy()
        label_text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        for line in label_text.splitlines():
            bbox = parse_yolo_bbox(line)
            if bbox is None:
                continue
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2.0) * vis.shape[1])
            y1 = int((cy - bh / 2.0) * vis.shape[0])
            x2 = int((cx + bw / 2.0) * vis.shape[1])
            y2 = int((cy + bh / 2.0) * vis.shape[0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        tag = f"{row['source_stage']} {row['target_color'] if truthy(row['has_target']) else row['target_type']}"
        cv2.putText(vis, tag[:24], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(cv2.resize(vis, (160, 160), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return
    cols = 6
    rows_count = math.ceil(len(thumbs) / cols)
    canvas = np.full((rows_count * 160, cols * 160, 3), 32, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        canvas[y : y + 160, x : x + 160] = thumb
    cv2.imwrite(str(out_dir / "preview_grid_final.jpg"), canvas)


def count_by(rows: list[dict[str, str]], key: str) -> Counter:
    return Counter(row.get(key, "none") or "none" for row in rows)


def split_counts(rows: list[dict[str, str]], key: str) -> dict[str, Counter]:
    result: dict[str, Counter] = {}
    for split in ("train", "val", "test"):
        result[split] = Counter(row.get(key, "none") or "none" for row in rows if row["final_split"] == split)
    return result


def duplicate_warnings(rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    image_groups: dict[str, list[str]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        image_groups[row["image_hash"]].append(row["final_image_name"])
        pair_groups[(row["image_hash"], row["label_hash"])].append(row["final_image_name"])
    dup_images = {h: names for h, names in image_groups.items() if len(names) > 1}
    dup_pairs = {h: names for h, names in pair_groups.items() if len(names) > 1}
    if dup_images:
        warnings.append(f"Duplicate image hash bulundu: {len(dup_images)} grup")
        for names in list(dup_images.values())[:8]:
            warnings.append(f"  image duplicate: {', '.join(names[:6])}")
    if dup_pairs:
        warnings.append(f"Duplicate image+label hash bulundu: {len(dup_pairs)} grup")
        for names in list(dup_pairs.values())[:8]:
            warnings.append(f"  image+label duplicate: {', '.join(names[:6])}")
    return warnings


def run_sanity_checks(out_dir: Path, rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "0: target_square" not in yaml_text:
        errors.append("data.yaml tek sinifli target_square degil")

    for row in rows:
        split = row["final_split"]
        image_path = out_dir / "images" / split / row["final_image_name"]
        label_path = out_dir / "labels" / split / f"{Path(row['final_image_name']).stem}.txt"
        if not image_path.exists():
            errors.append(f"image eksik: {image_path}")
            continue
        if not label_path.exists():
            errors.append(f"label eksik: {label_path}")
            continue
        label_text = label_path.read_text(encoding="utf-8").strip()
        if truthy(row["has_target"]) and not label_text:
            errors.append(f"has_target=true ama final label bos: {label_path.name}")
        if not truthy(row["has_target"]) and label_text:
            errors.append(f"has_target=false ama final label dolu: {label_path.name}")
        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"label format bozuk: {label_path.name}")
                continue
            if parts[0] != "0":
                errors.append(f"class id 0 degil: {label_path.name} -> {parts[0]}")
            try:
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"label sayisal degil: {label_path.name}")
                continue
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"bbox 0-1 disinda: {label_path.name}")

    warnings.extend(duplicate_warnings(rows))
    return errors, warnings


def report_counter(title: str, counter: Counter) -> list[str]:
    lines = [title]
    for key, value in counter.most_common():
        lines.append(f"  {key}: {value}")
    return lines


def report_split_counter(title: str, counters: dict[str, Counter]) -> list[str]:
    lines = [title]
    for split in ("train", "val", "test"):
        lines.append(f"  [{split}]")
        for key, value in counters[split].most_common():
            lines.append(f"    {key}: {value}")
    return lines


def write_sanity_report(
    out_dir: Path,
    rows: list[dict[str, str]],
    source_warnings: list[str],
    sampling_warnings: list[str],
    label_stats: Counter,
    sanity_errors: list[str],
    sanity_warnings: list[str],
) -> None:
    lines: list[str] = []
    lines.append("ASFLY final dataset sanity report")
    lines.append("=" * 38)
    lines.append("")
    lines.append(f"Total images: {len(rows)}")
    lines.extend(report_counter("Split counts:", count_by(rows, "final_split")))
    lines.extend(report_counter("Stage counts:", count_by(rows, "source_stage")))
    lines.extend(report_counter("Target color counts:", count_by(rows, "target_color")))
    lines.extend(report_counter("Has target counts:", count_by(rows, "has_target")))
    lines.extend(report_counter("Background type counts:", count_by(rows, "background_type")))
    lines.extend(report_counter("Negative type counts:", count_by(rows, "negative_type")))
    lines.extend(report_counter("Blur level counts:", count_by(rows, "blur_level")))
    lines.extend(report_counter("Lighting type counts:", count_by(rows, "lighting_type")))
    lines.extend(report_counter("Shadow level counts:", count_by(rows, "shadow_level")))
    lines.extend(report_counter("Glare level counts:", count_by(rows, "glare_level")))
    lines.extend(report_counter("Exposure level counts:", count_by(rows, "exposure_level")))
    lines.extend(report_counter("Color shift level counts:", count_by(rows, "color_shift_level")))
    lines.extend(report_counter("BBox px size counts:", count_by(rows, "bbox_px_size")))
    lines.append("")
    lines.extend(report_split_counter("Split stage distribution:", split_counts(rows, "source_stage")))
    lines.extend(report_split_counter("Split target_color distribution:", split_counts(rows, "target_color")))
    lines.extend(report_split_counter("Split has_target distribution:", split_counts(rows, "has_target")))
    lines.extend(report_split_counter("Split blur_level distribution:", split_counts(rows, "blur_level")))
    lines.extend(report_split_counter("Split lighting_type distribution:", split_counts(rows, "lighting_type")))
    lines.append("")
    lines.append("Label normalization:")
    lines.append(f"  converted label lines to class 0: {label_stats.get('converted_label_lines', 0)}")
    lines.append(f"  target_with_empty_label skipped: {label_stats.get('target_with_empty_label', 0)}")
    lines.append(f"  empty_with_label skipped: {label_stats.get('empty_with_label', 0)}")
    lines.append("")
    lines.append("Warnings:")
    warnings = source_warnings + sampling_warnings + sanity_warnings
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  none")
    lines.append("")
    lines.append("Errors:")
    if sanity_errors:
        for error in sanity_errors:
            lines.append(f"  - {error}")
    else:
        lines.append("  none")
    lines.append("")
    (out_dir / "sanity_report.txt").write_text("\n".join(lines), encoding="utf-8")


def cleanup_dry_run(paths: list[Path]) -> list[str]:
    report: list[str] = []
    for path in paths:
        lowered = path.name.lower()
        if any(token in lowered for token in ("preview", "debug", "annotated", "vis", "temp", "cache", "__pycache__")):
            report.append(f"would exclude from training source: {path}")
    return report


def default_output_name(mode: str) -> str:
    if mode == "debug":
        return "ds_final_target_square_debug"
    return "ds_final_target_square"


def build_final_dataset(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root) if args.output_root else datasets_root
    out_dir = output_root / (args.output_name or default_output_name(args.mode))

    samples_by_stage, source_warnings = collect_all_stage_samples(datasets_root)
    valid_by_stage: dict[str, list[Sample]] = {}
    label_stats: Counter = Counter()
    for stage_name, samples in samples_by_stage.items():
        valid, warnings, stats = filter_valid_training_images(samples)
        source_warnings.extend(warnings)
        label_stats.update(stats)
        valid_by_stage[stage_name] = valid

    target_counts = mode_target_counts(args.mode, args.max_per_stage)
    selected, sampling_warnings = sample_by_stage_ratio(valid_by_stage, target_counts, rng)
    if not selected:
        raise RuntimeError("Final dataset icin hic gecerli sample bulunamadi")

    create_final_split(selected, rng)
    assign_final_names(selected)
    ensure_output_dir(out_dir, overwrite=args.overwrite)
    for sample in selected:
        copy_image_and_label(sample, out_dir)

    rows = merge_metadata(selected)
    write_metadata(out_dir, rows)
    write_data_yaml(out_dir)
    write_readme(out_dir, args.mode, target_counts)
    create_final_preview(out_dir, rows)
    sanity_errors, sanity_warnings = run_sanity_checks(out_dir, rows)
    write_sanity_report(out_dir, rows, source_warnings, sampling_warnings, label_stats, sanity_errors, sanity_warnings)

    if sanity_errors:
        raise RuntimeError("Final dataset sanity check hatalari var. sanity_report.txt dosyasina bak.")

    if args.cleanup_dry_run:
        cleanup_report = cleanup_dry_run(list(datasets_root.rglob("*")))
        (out_dir / "cleanup_dry_run.txt").write_text("\n".join(cleanup_report), encoding="utf-8")

    return out_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build final ASFLY single-class YOLO dataset")
    ap.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT), help="Root folder containing stage datasets")
    ap.add_argument("--output-root", default="", help="Root folder for final dataset; defaults to --datasets-root")
    ap.add_argument("--mode", choices=["debug", "mini_final", "full_final"], default="debug")
    ap.add_argument("--max-per-stage", type=int, default=50, help="Only for --mode debug")
    ap.add_argument("--output-name", default="", help="Output dataset folder name under datasets root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="Recreate output folder if it already exists")
    ap.add_argument("--cleanup-dry-run", action="store_true", help="Write cleanup_dry_run.txt without deleting anything")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = build_final_dataset(args)
    print(f"Final dataset ready: {out_dir}")


if __name__ == "__main__":
    main()
