"""
Single-class curriculum dataset generator for ASFLY IHA target detection.

YOLO class:
  0: target_square

Real target color is stored only in metadata.csv:
  red  -> 1x1 m target square
  blue -> 2x2 m target square

Outputs:
  datasets/ds_stage1_shape_basic/
  datasets/ds_stage2_field_clean/
  datasets/ds_stage3_field_blur/
  datasets/ds_stage4_field_sun_shadow/

Each dataset uses:
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
  data.yaml
  metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    cv2.setLogLevel(2)
except Exception:
    pass

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass


IMG_SIZE = 640
CLS_TARGET = 0

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BG_DIR = SCRIPT_DIR.parent / "backgrounds"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "datasets"

STAGE_DIRS = {
    1: "ds_stage1_shape_basic",
    2: "ds_stage2_field_clean",
    3: "ds_stage3_field_blur",
    4: "ds_stage4_field_sun_shadow",
}

STAGE2_V2_DIR = "ds_stage2_field_clean_v2_preview"
STAGE3_V2_DIR = "ds_stage3_field_blur_clean_motion_v2_preview"
STAGE4_V2_DIR = "ds_stage4_field_sun_shadow_controlled_v2_preview"

STAGE_NAMES = {
    1: "shape_basic",
    2: "field_clean",
    3: "field_blur",
    4: "field_sun_shadow",
}

METADATA_FIELDS = [
    "image_name",
    "split",
    "stage",
    "has_target",
    "target_color",
    "target_type",
    "background_type",
    "negative_type",
    "augmentation_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_px_size",
    "rotation_deg",
    "perspective_level",
    "lighting_level",
    "notes",
]

STAGE2_METADATA_FIELDS = [
    "image_name",
    "split",
    "stage",
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
    "lighting_level",
    "notes",
]

STAGE3_METADATA_FIELDS = [
    "image_name",
    "split",
    "stage",
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
    "lighting_level",
    "notes",
]

STAGE4_METADATA_FIELDS = [
    "image_name",
    "split",
    "stage",
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
    "lighting_type",
    "lighting_level",
    "shadow_level",
    "glare_level",
    "exposure_level",
    "color_shift_level",
    "notes",
]


@dataclass
class SampleSpec:
    has_target: bool
    target_color: str
    target_type: str
    augmentation_type: str
    background_type: str = "unknown"
    negative_type: str = "none"
    blur_type: str = "none"
    blur_level: str = "none"
    motion_blur_length: int = 0
    motion_blur_angle: float = 0.0
    sun_level: str = "normal"
    glare_level: str = "none"
    shadow_level: str = "none"
    lighting_type: str = "normal"
    lighting_level: str = "normal"
    exposure_level: str = "normal"
    color_shift_level: str = "none"
    disaster_level: str = "none"
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 0
    bbox_h: int = 0
    bbox_px_size: int = 0
    rotation_deg: float = 0.0
    perspective_level: float = 0.0
    notes: str = ""


def read_image(path: Path) -> np.ndarray | None:
    try:
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def infer_background_type(path: Path) -> str:
    name = path.name.lower()
    if any(k in name for k in ("dry", "steppe", "bozkir", "hay")):
        return "dry_grass"
    if any(k in name for k in ("grass", "cim", "field", "green")):
        return "grass"
    if any(k in name for k in ("soil", "earth", "toprak", "dirt", "ploughed")):
        return "soil"
    if any(k in name for k in ("asphalt", "road")):
        return "asphalt"
    return "field_mixed"


def load_backgrounds(bg_dir: Path) -> list[tuple[Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif", ".tif", ".tiff"}
    paths = [p for p in bg_dir.iterdir() if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    readable: list[tuple[Path, str]] = []
    blocked = 0
    for path in paths:
        if read_image(path) is None:
            blocked += 1
            continue
        readable.append((path, infer_background_type(path)))
    if not readable:
        print(f"WARNING: {len(paths)} background file(s) found but none are readable. Synthetic field fallback will be used.")
        return []
    if blocked:
        print(f"WARNING: skipped {blocked} unreadable background file(s).")
    return readable


def make_plain_background(stage: int) -> tuple[np.ndarray, str]:
    if stage == 1 and random.random() < 0.70:
        base = random.randint(178, 238)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
        noise = np.random.normal(0, random.uniform(1, 5), img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8), "plain"

    color = random.choice([(64, 112, 66), (92, 124, 76), (82, 92, 58), (85, 80, 65), (110, 105, 88)])
    img = np.full((IMG_SIZE, IMG_SIZE, 3), color, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(4, 14), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img, "simple_field"


def make_stage1_background() -> tuple[np.ndarray, str]:
    bg_type = random.choices(
        ["plain_white", "plain_light_gray", "plain_dark_gray", "gray_noise", "matte_texture"],
        weights=[18, 30, 18, 22, 12],
    )[0]

    if bg_type == "plain_white":
        base = random.randint(226, 248)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
    elif bg_type == "plain_light_gray":
        base = random.randint(176, 224)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
    elif bg_type == "plain_dark_gray":
        base = random.randint(74, 132)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
    elif bg_type == "gray_noise":
        base = random.randint(126, 214)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
        noise = np.random.normal(0, random.uniform(2.0, 7.0), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        base = random.randint(116, 204)
        img = np.full((IMG_SIZE, IMG_SIZE, 3), (base, base, base), dtype=np.uint8)
        low = np.random.normal(0, random.uniform(3, 8), (64, 64, 1)).astype(np.float32)
        low = cv2.resize(low, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        if low.ndim == 2:
            low = low[:, :, None]
        tint = np.array(random.choice([(1.0, 1.0, 1.0), (0.96, 1.0, 1.02), (1.02, 1.0, 0.96)]), dtype=np.float32)
        img = np.clip(img.astype(np.float32) * tint + low, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img, bg_type


def get_background(backgrounds: list[tuple[Path, str]], stage: int) -> tuple[np.ndarray, str]:
    if stage == 1:
        return make_stage1_background()

    if not backgrounds:
        return make_plain_background(stage)

    if stage == 1 and random.random() < 0.55:
        return make_plain_background(stage)

    for _ in range(20):
        path, bg_type = random.choice(backgrounds)
        img = read_image(path)
        if img is not None:
            break
    else:
        return make_plain_background(stage)

    h, w = img.shape[:2]
    if w < IMG_SIZE or h < IMG_SIZE:
        scale = max(IMG_SIZE / max(1, w), IMG_SIZE / max(1, h)) * random.uniform(1.04, 1.30)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    x = random.randint(0, max(0, w - IMG_SIZE))
    y = random.randint(0, max(0, h - IMG_SIZE))
    crop = img[y : y + IMG_SIZE, x : x + IMG_SIZE].copy()
    alpha = random.uniform(0.92, 1.10)
    beta = random.randint(-8, 18)
    return cv2.convertScaleAbs(crop, alpha=alpha, beta=beta), bg_type


def crop_background_image(path: Path, bg_type: str) -> tuple[np.ndarray, str] | None:
    img = read_image(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if w < IMG_SIZE or h < IMG_SIZE:
        scale = max(IMG_SIZE / max(1, w), IMG_SIZE / max(1, h)) * random.uniform(1.02, 1.18)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    x = random.randint(0, max(0, w - IMG_SIZE))
    y = random.randint(0, max(0, h - IMG_SIZE))
    crop = img[y : y + IMG_SIZE, x : x + IMG_SIZE].copy()
    return crop, bg_type


def is_stage2_background_ok(img: np.ndarray, bg_type: str) -> bool:
    if bg_type == "asphalt" and random.random() > 0.12:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_v = float(gray.mean())
    very_dark_ratio = float((gray < 35).sum()) / float(gray.size)
    very_bright_ratio = float((gray > 235).sum()) / float(gray.size)
    if mean_v < 42 or very_dark_ratio > 0.28:
        return False
    if very_bright_ratio > 0.22:
        return False
    return True


def get_stage2_v2_background(backgrounds: list[tuple[Path, str]]) -> tuple[np.ndarray, str]:
    preferred = [item for item in backgrounds if item[1] in {"dry_grass", "grass", "soil", "field_mixed"}]
    asphalt = [item for item in backgrounds if item[1] == "asphalt"]
    pool = preferred + (asphalt if random.random() < 0.10 else [])

    if not pool:
        return make_plain_background(2)

    for _ in range(40):
        path, bg_type = random.choice(pool)
        cropped = crop_background_image(path, bg_type)
        if cropped is None:
            continue
        img, bg_type = cropped
        if not is_stage2_background_ok(img, bg_type):
            continue
        alpha = random.uniform(0.95, 1.08)
        beta = random.randint(-6, 12)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), bg_type

    return make_plain_background(2)


def rotated_rect_points(cx: int, cy: int, w: int, h: int, angle_deg: float, perspective: float = 0.0) -> np.ndarray:
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    if perspective > 0:
        pts += np.random.uniform(-w * perspective, w * perspective, pts.shape).astype(np.float32)
    rad = math.radians(angle_deg)
    rot = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=np.float32)
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return np.round(pts).astype(np.int32)


def bbox_from_pts(pts: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(IMG_SIZE - 1, int(x2)),
        min(IMG_SIZE - 1, int(y2)),
    )


def yolo_label(x1: int, y1: int, x2: int, y2: int) -> str:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{CLS_TARGET} {cx / IMG_SIZE:.6f} {cy / IMG_SIZE:.6f} {bw / IMG_SIZE:.6f} {bh / IMG_SIZE:.6f}"


def sample_target_side(stage: int, color: str, size_bucket: str | None = None) -> int:
    # Blue target is 2m, red target is 1m. Pixel sizes are deliberately practical for YOLOv8n.
    if stage == 1:
        bucket = size_bucket or random.choices(["medium_large", "small", "very_small"], weights=[70, 20, 10])[0]
        if color == "red":
            ranges = {
                "medium_large": (62, 176),
                "small": (42, 61),
                "very_small": (30, 41),
            }
        else:
            ranges = {
                "medium_large": (90, 248),
                "small": (62, 89),
                "very_small": (44, 61),
            }
        lo, hi = ranges[bucket]
        return random.randint(lo, hi)

    if color == "red":
        ranges = {
            1: (34, 170),
            2: (20, 155),
            3: (18, 150),
            4: (18, 145),
        }
    else:
        ranges = {
            1: (54, 240),
            2: (32, 235),
            3: (28, 220),
            4: (28, 210),
        }
    lo, hi = ranges[stage]
    if stage >= 2 and random.random() < 0.20:
        hi = int((lo + hi) * 0.55)
    return random.randint(lo, hi)


def sample_stage2_v2_target_side(color: str, size_bucket: str) -> int:
    if color == "red":
        ranges = {
            "medium": (58, 116),
            "large": (118, 168),
            "small": (36, 56),
            "very_small": (26, 35),
        }
    else:
        ranges = {
            "medium": (88, 172),
            "large": (176, 252),
            "small": (54, 86),
            "very_small": (38, 53),
        }
    lo, hi = ranges[size_bucket]
    return random.randint(lo, hi)


def place_target(side: int) -> tuple[int, int]:
    margin = max(24, int(side * 0.70))
    return (
        random.randint(margin, IMG_SIZE - margin),
        random.randint(margin, IMG_SIZE - margin),
    )


def draw_target_square(
    img: np.ndarray,
    stage: int,
    color_name: str,
    size_bucket: str | None = None,
) -> tuple[np.ndarray, str, int, tuple[int, int, int, int], float, float]:
    side = sample_target_side(stage, color_name, size_bucket)
    cx, cy = place_target(side)
    angle = random.uniform(-8, 8) if stage == 1 else random.uniform(-28, 28)
    if stage >= 2 and random.random() < 0.22:
        angle = random.uniform(-45, 45)
    perspective = random.uniform(0.0, 0.012) if stage == 1 else random.uniform(0.00, 0.055)
    pts = rotated_rect_points(cx, cy, side, side, angle, perspective)

    if color_name == "red":
        base = np.array([random.randint(0, 35), random.randint(0, 55), random.randint(180, 255)], dtype=np.uint8)
    else:
        base = np.array([random.randint(170, 255), random.randint(25, 100), random.randint(0, 55)], dtype=np.uint8)

    light = random.uniform(0.86, 1.14)
    color = tuple(np.clip(base.astype(np.float32) * light, 0, 255).astype(np.uint8).tolist())
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    if stage != 1:
        edge = tuple(int(max(0, c - random.randint(25, 55))) for c in color)
        cv2.polylines(img, [pts], True, edge, max(1, side // 85), lineType=cv2.LINE_AA)

    x1, y1, x2, y2 = bbox_from_pts(pts)
    return img, yolo_label(x1, y1, x2, y2), max(x2 - x1, y2 - y1), (x1, y1, x2, y2), angle, perspective


def draw_stage2_v2_target_square(
    img: np.ndarray,
    color_name: str,
    size_bucket: str,
) -> tuple[np.ndarray, str, int, tuple[int, int, int, int], float, float]:
    side = sample_stage2_v2_target_side(color_name, size_bucket)
    cx, cy = place_target(side)
    angle = random.uniform(-24, 24)
    if random.random() < 0.10:
        angle = random.uniform(-32, 32)
    perspective = random.uniform(0.00, 0.032)
    pts = rotated_rect_points(cx, cy, side, side, angle, perspective)

    if color_name == "red":
        base = np.array([random.randint(0, 32), random.randint(0, 58), random.randint(188, 255)], dtype=np.uint8)
    else:
        base = np.array([random.randint(178, 255), random.randint(28, 105), random.randint(0, 58)], dtype=np.uint8)

    light = random.uniform(0.90, 1.10)
    color = tuple(np.clip(base.astype(np.float32) * light, 0, 255).astype(np.uint8).tolist())
    cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    edge = tuple(int(max(0, c - random.randint(18, 42))) for c in color)
    cv2.polylines(img, [pts], True, edge, max(1, side // 105), lineType=cv2.LINE_AA)

    x1, y1, x2, y2 = bbox_from_pts(pts)
    return img, yolo_label(x1, y1, x2, y2), max(x2 - x1, y2 - y1), (x1, y1, x2, y2), angle, perspective


def apply_motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    rad = math.radians(angle)
    dx = math.cos(rad) * c
    dy = math.sin(rad) * c
    cv2.line(kernel, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
    s = kernel.sum()
    if s <= 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def add_noise(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    std = random.uniform(lo, hi)
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, q_min: int, q_max: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(q_min, q_max)])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def color_light_jitter(img: np.ndarray, hard: bool = False) -> np.ndarray:
    if hard:
        alpha = random.uniform(0.72, 1.42)
        beta = random.randint(-45, 55)
        sat = random.randint(-34, 26)
    else:
        alpha = random.uniform(0.86, 1.20)
        beta = random.randint(-18, 26)
        sat = random.randint(-16, 18)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-8, 12), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def soft_shadow_light(img: np.ndarray, level: str) -> np.ndarray:
    h, w = img.shape[:2]
    if level == "shadow":
        alpha_rng, beta_rng, strength_rng = (0.82, 1.04), (-24, 8), (0.08, 0.22)
    elif level == "sunny":
        alpha_rng, beta_rng, strength_rng = (1.02, 1.24), (4, 32), (0.03, 0.12)
    else:
        alpha_rng, beta_rng, strength_rng = (0.88, 1.16), (-16, 20), (0.04, 0.16)

    out = cv2.convertScaleAbs(img, alpha=random.uniform(*alpha_rng), beta=random.randint(*beta_rng)).astype(np.float32)
    axis = random.choice(["x", "y"])
    if axis == "x":
        grad = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    else:
        grad = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
    if random.random() < 0.5:
        grad = 1.0 - grad
    grad = cv2.GaussianBlur(grad, (0, 0), random.uniform(35, 95))
    if level == "sunny":
        out *= 1.0 + random.uniform(*strength_rng) * grad[:, :, None]
    else:
        out *= 1.0 - random.uniform(*strength_rng) * grad[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def partial_glare(img: np.ndarray, bbox_px_size: int, disaster: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    x = random.randint(0, w - 1)
    y = random.randint(0, h - 1)
    if disaster:
        r = int(np.clip(max(30, bbox_px_size) * random.uniform(0.55, 1.35), 24, 130))
        amount = random.uniform(0.28, 0.72)
    else:
        r = int(np.clip(max(24, bbox_px_size) * random.uniform(0.18, 0.55), 14, 70))
        amount = random.uniform(0.08, 0.22)
    cv2.circle(overlay, (x, y), r, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), r * random.uniform(1.2, 2.6))
    out = img.astype(np.float32) + overlay.astype(np.float32) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_negative_object(img: np.ndarray, stage: int) -> tuple[np.ndarray, str]:
    kind = random.choices(
        ["thin_marker", "small_sign", "faded_patch", "circle", "triangle", "cloth"],
        weights=[0.30, 0.25, 0.18, 0.11, 0.08, 0.08],
    )[0]
    cx, cy = random.randint(55, IMG_SIZE - 55), random.randint(55, IMG_SIZE - 55)
    color = random.choice([
        (random.randint(0, 55), random.randint(0, 90), random.randint(120, 230)),
        (random.randint(120, 230), random.randint(25, 120), random.randint(0, 90)),
        (random.randint(160, 230), random.randint(160, 230), random.randint(160, 230)),
    ])

    if kind == "thin_marker":
        pts = rotated_rect_points(cx, cy, random.randint(42, 140), random.randint(5, 16), random.uniform(0, 180))
        cv2.fillPoly(img, [pts], color)
    elif kind == "small_sign":
        w = random.randint(38, 115)
        h = int(np.clip(w * random.choice([0.36, 0.50, 0.65, 1.55]), 16, 125))
        pts = rotated_rect_points(cx, cy, w, h, random.uniform(-35, 35))
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, tuple(max(0, c - 45) for c in color), 1)
    elif kind == "faded_patch":
        cv2.ellipse(img, (cx, cy), (random.randint(14, 58), random.randint(8, 36)), random.uniform(0, 180), 0, 360, color, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif kind == "circle":
        cv2.circle(img, (cx, cy), random.randint(8, 34), color, -1)
    elif kind == "triangle":
        w, h = random.randint(15, 55), random.randint(15, 55)
        pts = np.array([[cx, cy - h], [cx - w, cy + h], [cx + w, cy + h]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    else:
        pts = rotated_rect_points(cx, cy, random.randint(32, 100), random.randint(18, 65), random.uniform(0, 180), 0.12)
        cv2.fillPoly(img, [pts], color)

    return img, f"negative_{kind}"


def stage_recipe(stage: int) -> tuple[str, str]:
    if stage == 1:
        target = random.choices(["red", "blue", "empty"], weights=[45, 45, 10])[0]
        aug = random.choices(["clean", "mild_light", "mild_noise"], weights=[68, 22, 10])[0]
    elif stage == 2:
        target = random.choices(["red", "blue", "empty", "negative"], weights=[40, 40, 15, 5])[0]
        aug = random.choices(["clean", "basic_light"], weights=[70, 30])[0]
    elif stage == 3:
        target = random.choices(["red", "blue", "empty", "negative"], weights=[38, 37, 15, 10])[0]
        aug = random.choices(["clean", "light_blur", "motion_blur", "basic_light"], weights=[35, 25, 15, 25])[0]
    else:
        target = random.choices(["red", "blue", "empty", "negative"], weights=[36, 36, 18, 10])[0]
        aug = random.choices(
            ["clean", "shadow", "sunny", "partial_glare", "blur_light", "disaster"],
            weights=[30, 20, 20, 10, 15, 5],
        )[0]
    return target, aug


def build_stage1_plan(count: int) -> list[dict[str, str]]:
    red_count = int(round(count * 0.45))
    blue_count = int(round(count * 0.45))
    empty_count = max(0, count - red_count - blue_count)

    targets = ["red"] * red_count + ["blue"] * blue_count + ["empty"] * empty_count
    random.shuffle(targets)

    target_count = red_count + blue_count
    medium_count = int(round(target_count * 0.70))
    small_count = int(round(target_count * 0.20))
    very_small_count = max(0, target_count - medium_count - small_count)
    size_buckets = (
        ["medium_large"] * medium_count +
        ["small"] * small_count +
        ["very_small"] * very_small_count
    )
    random.shuffle(size_buckets)

    aug_clean = int(round(count * 0.68))
    aug_light = int(round(count * 0.22))
    aug_noise = max(0, count - aug_clean - aug_light)
    augmentations = ["clean"] * aug_clean + ["mild_light"] * aug_light + ["mild_noise"] * aug_noise
    random.shuffle(augmentations)

    plan: list[dict[str, str]] = []
    size_i = 0
    for i, target in enumerate(targets):
        size_bucket = "none"
        if target in ("red", "blue"):
            size_bucket = size_buckets[size_i]
            size_i += 1
        plan.append({
            "target": target,
            "augmentation": augmentations[i],
            "size_bucket": size_bucket,
        })
    return plan


def build_stage2_v2_plan(count: int) -> list[dict[str, str]]:
    red_count = int(round(count * 0.40))
    blue_count = int(round(count * 0.40))
    empty_count = int(round(count * 0.15))
    negative_count = max(0, count - red_count - blue_count - empty_count)

    targets = (
        ["red"] * red_count +
        ["blue"] * blue_count +
        ["empty"] * empty_count +
        ["negative"] * negative_count
    )
    random.shuffle(targets)

    target_count = red_count + blue_count
    medium_count = int(round(target_count * 0.55))
    large_count = int(round(target_count * 0.23))
    small_count = int(round(target_count * 0.17))
    very_small_count = max(0, target_count - medium_count - large_count - small_count)
    size_buckets = (
        ["medium"] * medium_count +
        ["large"] * large_count +
        ["small"] * small_count +
        ["very_small"] * very_small_count
    )
    random.shuffle(size_buckets)

    clean_count = int(round(count * 0.64))
    mild_light_count = int(round(count * 0.25))
    mild_noise_count = int(round(count * 0.07))
    mild_color_count = max(0, count - clean_count - mild_light_count - mild_noise_count)
    augmentations = (
        ["clean"] * clean_count +
        ["mild_light"] * mild_light_count +
        ["mild_noise"] * mild_noise_count +
        ["mild_color"] * mild_color_count
    )
    random.shuffle(augmentations)

    plan: list[dict[str, str]] = []
    size_i = 0
    for i, target in enumerate(targets):
        size_bucket = "none"
        if target in ("red", "blue"):
            size_bucket = size_buckets[size_i]
            size_i += 1
        plan.append({
            "target": target,
            "augmentation": augmentations[i],
            "size_bucket": size_bucket,
        })
    return plan


def build_stage3_v2_plan(count: int) -> list[dict[str, str]]:
    clean_count = int(round(count * 0.45))
    light_motion_count = int(round(count * 0.25))
    medium_motion_count = int(round(count * 0.15))
    defocus_count = int(round(count * 0.05))
    empty_count = int(round(count * 0.05))
    negative_count = max(0, count - clean_count - light_motion_count - medium_motion_count - defocus_count - empty_count)
    positive_count = clean_count + light_motion_count + medium_motion_count + defocus_count
    red_count = positive_count // 2
    blue_count = positive_count - red_count

    positive_augs = (
        ["clean"] * clean_count +
        ["light_motion"] * light_motion_count +
        ["medium_motion"] * medium_motion_count +
        ["light_defocus"] * defocus_count
    )
    random.shuffle(positive_augs)

    target_count = red_count + blue_count
    medium_count = int(round(target_count * 0.52))
    large_count = int(round(target_count * 0.22))
    small_count = int(round(target_count * 0.18))
    very_small_count = max(0, target_count - medium_count - large_count - small_count)
    size_buckets = (
        ["medium"] * medium_count +
        ["large"] * large_count +
        ["small"] * small_count +
        ["very_small"] * very_small_count
    )
    random.shuffle(size_buckets)

    aug_size_pairs = list(zip(positive_augs, size_buckets))
    for i, (aug, bucket) in enumerate(aug_size_pairs):
        if aug != "medium_motion" or bucket not in {"small", "very_small"}:
            continue
        for j, (other_aug, other_bucket) in enumerate(aug_size_pairs):
            if other_aug != "medium_motion" and other_bucket in {"medium", "large"}:
                aug_size_pairs[i] = (other_aug, bucket)
                aug_size_pairs[j] = ("medium_motion", other_bucket)
                break
    positive_augs = [pair[0] for pair in aug_size_pairs]
    size_buckets = [pair[1] for pair in aug_size_pairs]

    targets = ["red"] * red_count + ["blue"] * blue_count
    random.shuffle(targets)

    plan: list[dict[str, str]] = []
    for i, target in enumerate(targets):
        plan.append({
            "target": target,
            "augmentation": positive_augs[i],
            "size_bucket": size_buckets[i],
        })

    for _ in range(empty_count):
        plan.append({
            "target": "empty",
            "augmentation": random.choices(["clean", "light_motion"], weights=[75, 25])[0],
            "size_bucket": "none",
        })
    for _ in range(negative_count):
        plan.append({
            "target": "negative",
            "augmentation": random.choices(["clean", "light_motion"], weights=[70, 30])[0],
            "size_bucket": "none",
        })

    random.shuffle(plan)
    return plan


def build_stage4_v2_plan(count: int) -> list[dict[str, str]]:
    clean_count = int(round(count * 0.30))
    shadow_count = int(round(count * 0.20))
    sunny_count = int(round(count * 0.20))
    glare_count = int(round(count * 0.10))
    blur_light_count = int(round(count * 0.10))
    empty_count = int(round(count * 0.05))
    negative_count = max(0, count - clean_count - shadow_count - sunny_count - glare_count - blur_light_count - empty_count)
    positive_count = clean_count + shadow_count + sunny_count + glare_count + blur_light_count

    red_count = positive_count // 2
    blue_count = positive_count - red_count
    targets = ["red"] * red_count + ["blue"] * blue_count
    random.shuffle(targets)

    augmentations = (
        ["clean"] * clean_count +
        ["shadow"] * shadow_count +
        ["sunny"] * sunny_count +
        ["partial_glare"] * glare_count +
        ["blur_light"] * blur_light_count
    )
    random.shuffle(augmentations)

    medium_count = int(round(positive_count * 0.50))
    large_count = int(round(positive_count * 0.24))
    small_count = int(round(positive_count * 0.18))
    very_small_count = max(0, positive_count - medium_count - large_count - small_count)
    size_buckets = (
        ["medium"] * medium_count +
        ["large"] * large_count +
        ["small"] * small_count +
        ["very_small"] * very_small_count
    )
    random.shuffle(size_buckets)

    aug_size_pairs = list(zip(augmentations, size_buckets))
    for i, (aug, bucket) in enumerate(aug_size_pairs):
        if aug not in {"partial_glare", "blur_light"} or bucket not in {"small", "very_small"}:
            continue
        for j, (other_aug, other_bucket) in enumerate(aug_size_pairs):
            if other_aug in {"clean", "shadow", "sunny"} and other_bucket in {"medium", "large"}:
                aug_size_pairs[i] = (other_aug, bucket)
                aug_size_pairs[j] = (aug, other_bucket)
                break
    augmentations = [pair[0] for pair in aug_size_pairs]
    size_buckets = [pair[1] for pair in aug_size_pairs]

    plan: list[dict[str, str]] = []
    for i, target in enumerate(targets):
        plan.append({
            "target": target,
            "augmentation": augmentations[i],
            "size_bucket": size_buckets[i],
        })

    for _ in range(empty_count):
        plan.append({
            "target": "empty",
            "augmentation": random.choices(["clean", "shadow", "sunny"], weights=[55, 25, 20])[0],
            "size_bucket": "none",
        })
    for _ in range(negative_count):
        plan.append({
            "target": "negative",
            "augmentation": random.choices(["clean", "shadow", "sunny"], weights=[55, 25, 20])[0],
            "size_bucket": "none",
        })

    random.shuffle(plan)
    return plan


def make_stage1_sample(plan: dict[str, str]) -> tuple[np.ndarray, list[str], SampleSpec]:
    img, bg_type = make_stage1_background()
    target = plan["target"]
    spec = SampleSpec(
        has_target=target in ("red", "blue"),
        target_color=target if target in ("red", "blue") else "none",
        target_type="square" if target in ("red", "blue") else "background",
        augmentation_type=plan["augmentation"],
        background_type=bg_type,
        notes=f"size_bucket={plan['size_bucket']}",
    )

    labels: list[str] = []
    if target in ("red", "blue"):
        img, label, bbox_size, bbox, angle, perspective = draw_target_square(img, 1, target, plan["size_bucket"])
        labels.append(label)
        x1, y1, x2, y2 = bbox
        spec.bbox_x = x1
        spec.bbox_y = y1
        spec.bbox_w = max(1, x2 - x1)
        spec.bbox_h = max(1, y2 - y1)
        spec.bbox_px_size = bbox_size
        spec.rotation_deg = angle
        spec.perspective_level = perspective

    img = apply_stage1_augmentation(img, spec)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)
        labels = [flip_label_x(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_x = IMG_SIZE - spec.bbox_x - spec.bbox_w
    if random.random() < 0.18:
        img = cv2.flip(img, 0)
        labels = [flip_label_y(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_y = IMG_SIZE - spec.bbox_y - spec.bbox_h

    return img, labels, spec


def apply_stage1_augmentation(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    if spec.augmentation_type == "clean":
        return img
    if spec.augmentation_type == "mild_light":
        spec.sun_level = "mild"
        return cv2.convertScaleAbs(
            img,
            alpha=random.uniform(0.92, 1.10),
            beta=random.randint(-10, 12),
        )
    if spec.augmentation_type == "mild_noise":
        spec.sun_level = "normal"
        return add_noise(img, 1.0, 4.0)
    return img


def apply_stage2_v2_augmentation(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    aug = spec.augmentation_type
    if aug == "clean":
        spec.lighting_level = "normal"
        return img
    if aug == "mild_light":
        spec.lighting_level = "mild"
        return cv2.convertScaleAbs(
            img,
            alpha=random.uniform(0.92, 1.10),
            beta=random.randint(-10, 14),
        )
    if aug == "mild_noise":
        spec.lighting_level = "normal"
        return add_noise(img, 1.0, 4.5)
    if aug == "mild_color":
        spec.lighting_level = "mild_color"
        out = cv2.convertScaleAbs(img, alpha=random.uniform(0.94, 1.08), beta=random.randint(-8, 12))
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-8, 10), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-6, 8), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def apply_stage3_v2_augmentation(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    aug = spec.augmentation_type
    side = max(1, spec.bbox_px_size)
    if aug == "clean":
        spec.blur_type = "none"
        spec.blur_level = "none"
        spec.lighting_level = "normal"
        return img
    if aug == "light_defocus":
        spec.blur_type = "defocus"
        spec.blur_level = "light"
        spec.lighting_level = "normal"
        return cv2.GaussianBlur(img, (3, 3), 0)
    if aug in {"light_motion", "medium_motion"}:
        spec.blur_level = "motion_controlled"
        spec.blur_type = "motion"
        spec.blur_level = "light" if aug == "light_motion" else "medium"
        spec.lighting_level = "normal"
        if "size_bucket=very_small" in spec.notes:
            max_len = 5
            spec.blur_level = "light"
        elif "size_bucket=small" in spec.notes:
            max_len = 7
            spec.blur_level = "light"
        elif side <= 40:
            max_len = 5
            spec.blur_level = "light"
        elif side <= 60:
            max_len = 7
            spec.blur_level = "light"
        elif side <= 120:
            max_len = 9 if aug == "light_motion" else 11
        else:
            max_len = 11 if aug == "light_motion" else 15
        choices = [k for k in [5, 7, 9, 11, 13, 15] if k <= max_len] or [5]
        length = random.choice(choices)
        angle = random.uniform(0, 180)
        spec.motion_blur_length = length
        spec.motion_blur_angle = angle
        return apply_motion_blur(img, length, angle)
    return img


def spec_size_bucket(spec: SampleSpec) -> str:
    for bucket in ("very_small", "small", "medium", "large"):
        if f"size_bucket={bucket}" in spec.notes:
            return bucket
    return "none"


def mild_color_shift(img: np.ndarray, spec: SampleSpec, max_sat_delta: int = 10) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    sat_delta = random.randint(-max_sat_delta, max_sat_delta)
    val_delta = random.randint(-6, 8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_delta, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_delta, 0, 255)
    if abs(sat_delta) >= 7:
        spec.color_shift_level = "light"
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_controlled_shadow(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    bucket = spec_size_bucket(spec)
    h, w = img.shape[:2]
    if bucket in {"very_small", "small"}:
        strength = random.uniform(0.08, 0.15)
        spec.shadow_level = "light"
        spec.lighting_level = "light"
    else:
        strength = random.uniform(0.14, 0.26)
        spec.shadow_level = "medium"
        spec.lighting_level = "medium"

    axis = random.choice(["x", "y"])
    if axis == "x":
        grad = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    else:
        grad = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
    if random.random() < 0.5:
        grad = 1.0 - grad
    grad = cv2.GaussianBlur(grad, (0, 0), random.uniform(55, 120))
    out = img.astype(np.float32) * (1.0 - strength * grad[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_controlled_sun(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    bucket = spec_size_bucket(spec)
    if bucket in {"very_small", "small"}:
        alpha = random.uniform(1.03, 1.10)
        beta = random.randint(3, 12)
        spec.lighting_level = "light"
    else:
        alpha = random.uniform(1.07, 1.18)
        beta = random.randint(6, 24)
        spec.lighting_level = "medium"
    spec.exposure_level = "mild_over"
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return mild_color_shift(out, spec, max_sat_delta=8)


def apply_controlled_glare(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    bucket = spec_size_bucket(spec)
    h, w = img.shape[:2]
    side = max(1, spec.bbox_px_size)
    if bucket in {"very_small", "small"}:
        radius = int(np.clip(side * random.uniform(0.10, 0.18), 8, 18))
        amount = random.uniform(0.08, 0.14)
        spec.glare_level = "light"
        spec.lighting_level = "light"
    else:
        radius = int(np.clip(side * random.uniform(0.18, 0.34), 14, 58))
        amount = random.uniform(0.12, 0.23)
        spec.glare_level = "medium"
        spec.lighting_level = "medium"

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if spec.has_target and random.random() < 0.55:
        x1, y1 = spec.bbox_x, spec.bbox_y
        x2, y2 = spec.bbox_x + spec.bbox_w, spec.bbox_y + spec.bbox_h
        if random.random() < 0.55:
            cx = random.choice([x1, x2]) + random.randint(-radius, radius)
            cy = random.randint(max(0, y1), min(h - 1, y2))
        else:
            cx = random.randint(max(0, x1 - radius * 2), min(w - 1, x2 + radius * 2))
            cy = random.randint(max(0, y1 - radius * 2), min(h - 1, y2 + radius * 2))
    else:
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

    cv2.circle(overlay, (int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))), radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), radius * random.uniform(1.1, 2.0))
    out = img.astype(np.float32) + overlay.astype(np.float32) * amount
    spec.exposure_level = "mild_over"
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_stage4_v2_augmentation(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    aug = spec.augmentation_type
    side = max(1, spec.bbox_px_size)
    if aug == "clean":
        spec.lighting_type = "normal"
        spec.lighting_level = "normal"
        spec.exposure_level = "normal"
        return img
    if aug == "shadow":
        spec.lighting_type = "shadow"
        return apply_controlled_shadow(img, spec)
    if aug == "sunny":
        spec.sun_level = "bright"
        spec.lighting_type = "sunny"
        return apply_controlled_sun(img, spec)
    if aug == "partial_glare":
        spec.lighting_type = "glare"
        img = apply_controlled_sun(img, spec) if random.random() < 0.35 else img
        return apply_controlled_glare(img, spec)
    if aug == "blur_light":
        spec.blur_type = "motion"
        spec.blur_level = "light"
        spec.lighting_type = "blur_light"
        bucket = spec_size_bucket(spec)
        if bucket in {"very_small", "small"}:
            max_len = 5
            spec.lighting_level = "light"
        elif side < 130:
            max_len = 9
            spec.lighting_level = "light"
        else:
            max_len = 11
            spec.lighting_level = "medium"
        choices = [k for k in [5, 7, 9, 11] if k <= max_len] or [5]
        length = random.choice(choices)
        angle = random.uniform(0, 180)
        spec.motion_blur_length = length
        spec.motion_blur_angle = angle
        img = apply_motion_blur(img, length, angle)
        if random.random() < 0.55:
            return apply_controlled_sun(img, spec)
        return apply_controlled_shadow(img, spec)
    return img


def draw_stage2_v2_negative_object(img: np.ndarray) -> tuple[np.ndarray, str]:
    kind = random.choices(
        ["red_long_panel", "blue_tarp_strip", "white_ground_mark", "colored_cloth", "round_object"],
        weights=[24, 24, 18, 20, 14],
    )[0]
    cx, cy = random.randint(70, IMG_SIZE - 70), random.randint(70, IMG_SIZE - 70)

    if kind == "red_long_panel":
        color = (random.randint(0, 35), random.randint(0, 65), random.randint(135, 220))
        pts = rotated_rect_points(cx, cy, random.randint(70, 170), random.randint(14, 38), random.uniform(-45, 45), 0.025)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    elif kind == "blue_tarp_strip":
        color = (random.randint(135, 235), random.randint(35, 115), random.randint(0, 65))
        pts = rotated_rect_points(cx, cy, random.randint(80, 190), random.randint(12, 42), random.uniform(0, 180), 0.045)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    elif kind == "white_ground_mark":
        color = (random.randint(185, 238), random.randint(185, 238), random.randint(185, 238))
        pts = rotated_rect_points(cx, cy, random.randint(70, 160), random.randint(5, 14), random.uniform(0, 180), 0.015)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    elif kind == "colored_cloth":
        color = random.choice([
            (random.randint(0, 60), random.randint(0, 85), random.randint(120, 220)),
            (random.randint(120, 225), random.randint(35, 115), random.randint(0, 80)),
        ])
        pts = rotated_rect_points(cx, cy, random.randint(45, 115), random.randint(22, 70), random.uniform(0, 180), 0.16)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)
    else:
        color = random.choice([
            (random.randint(0, 55), random.randint(0, 85), random.randint(130, 220)),
            (random.randint(130, 230), random.randint(35, 120), random.randint(0, 80)),
        ])
        axes = (random.randint(12, 40), random.randint(9, 32))
        cv2.ellipse(img, (cx, cy), axes, random.uniform(0, 180), 0, 360, color, -1, lineType=cv2.LINE_AA)

    return img, kind


def make_stage2_v2_sample(
    backgrounds: list[tuple[Path, str]],
    plan: dict[str, str],
) -> tuple[np.ndarray, list[str], SampleSpec]:
    img, bg_type = get_stage2_v2_background(backgrounds)
    target = plan["target"]
    spec = SampleSpec(
        has_target=target in ("red", "blue"),
        target_color=target if target in ("red", "blue") else "none",
        target_type="square" if target in ("red", "blue") else ("negative" if target == "negative" else "background"),
        augmentation_type=plan["augmentation"],
        background_type=bg_type,
        notes=f"size_bucket={plan['size_bucket']};augmentation={plan['augmentation']}",
    )

    labels: list[str] = []
    if target in ("red", "blue"):
        img, label, bbox_size, bbox, angle, perspective = draw_stage2_v2_target_square(img, target, plan["size_bucket"])
        labels.append(label)
        x1, y1, x2, y2 = bbox
        spec.bbox_x = x1
        spec.bbox_y = y1
        spec.bbox_w = max(1, x2 - x1)
        spec.bbox_h = max(1, y2 - y1)
        spec.bbox_px_size = bbox_size
        spec.rotation_deg = angle
        spec.perspective_level = perspective
    elif target == "negative":
        n = 1 if random.random() < 0.90 else 2
        negative_notes = []
        for _ in range(n):
            img, negative_type = draw_stage2_v2_negative_object(img)
            negative_notes.append(negative_type)
        spec.negative_type = "+".join(negative_notes)
        spec.notes += f";negative={spec.negative_type}"

    img = apply_stage2_v2_augmentation(img, spec)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)
        labels = [flip_label_x(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_x = IMG_SIZE - spec.bbox_x - spec.bbox_w
    if random.random() < 0.12:
        img = cv2.flip(img, 0)
        labels = [flip_label_y(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_y = IMG_SIZE - spec.bbox_y - spec.bbox_h

    return img, labels, spec


def make_stage3_v2_sample(
    backgrounds: list[tuple[Path, str]],
    plan: dict[str, str],
) -> tuple[np.ndarray, list[str], SampleSpec]:
    img, bg_type = get_stage2_v2_background(backgrounds)
    target = plan["target"]
    spec = SampleSpec(
        has_target=target in ("red", "blue"),
        target_color=target if target in ("red", "blue") else "none",
        target_type="square" if target in ("red", "blue") else ("negative" if target == "negative" else "background"),
        augmentation_type=plan["augmentation"],
        background_type=bg_type,
        notes=f"size_bucket={plan['size_bucket']};augmentation={plan['augmentation']}",
    )

    labels: list[str] = []
    if target in ("red", "blue"):
        img, label, bbox_size, bbox, angle, perspective = draw_stage2_v2_target_square(img, target, plan["size_bucket"])
        labels.append(label)
        x1, y1, x2, y2 = bbox
        spec.bbox_x = x1
        spec.bbox_y = y1
        spec.bbox_w = max(1, x2 - x1)
        spec.bbox_h = max(1, y2 - y1)
        spec.bbox_px_size = bbox_size
        spec.rotation_deg = angle
        spec.perspective_level = perspective
    elif target == "negative":
        n = 1 if random.random() < 0.88 else 2
        negative_notes = []
        for _ in range(n):
            img, negative_type = draw_stage2_v2_negative_object(img)
            negative_notes.append(negative_type)
        spec.negative_type = "+".join(negative_notes)
        spec.notes += f";negative={spec.negative_type}"

    img = apply_stage3_v2_augmentation(img, spec)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)
        labels = [flip_label_x(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_x = IMG_SIZE - spec.bbox_x - spec.bbox_w
    if random.random() < 0.12:
        img = cv2.flip(img, 0)
        labels = [flip_label_y(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_y = IMG_SIZE - spec.bbox_y - spec.bbox_h

    return img, labels, spec


def make_stage4_v2_sample(
    backgrounds: list[tuple[Path, str]],
    plan: dict[str, str],
) -> tuple[np.ndarray, list[str], SampleSpec]:
    img, bg_type = get_stage2_v2_background(backgrounds)
    target = plan["target"]
    if target in ("red", "blue") and plan["size_bucket"] in {"very_small", "small"} and plan["augmentation"] in {"partial_glare", "blur_light"}:
        plan = dict(plan)
        plan["augmentation"] = random.choice(["clean", "shadow", "sunny"])
    spec = SampleSpec(
        has_target=target in ("red", "blue"),
        target_color=target if target in ("red", "blue") else "none",
        target_type="square" if target in ("red", "blue") else ("negative" if target == "negative" else "background"),
        augmentation_type=plan["augmentation"],
        background_type=bg_type,
        notes=f"size_bucket={plan['size_bucket']};augmentation={plan['augmentation']}",
    )

    labels: list[str] = []
    if target in ("red", "blue"):
        img, label, bbox_size, bbox, angle, perspective = draw_stage2_v2_target_square(img, target, plan["size_bucket"])
        labels.append(label)
        x1, y1, x2, y2 = bbox
        spec.bbox_x = x1
        spec.bbox_y = y1
        spec.bbox_w = max(1, x2 - x1)
        spec.bbox_h = max(1, y2 - y1)
        spec.bbox_px_size = bbox_size
        spec.rotation_deg = angle
        spec.perspective_level = perspective
    elif target == "negative":
        n = 1 if random.random() < 0.88 else 2
        negative_notes = []
        for _ in range(n):
            img, negative_type = draw_stage2_v2_negative_object(img)
            negative_notes.append(negative_type)
        spec.negative_type = "+".join(negative_notes)
        spec.notes += f";negative={spec.negative_type}"

    img = apply_stage4_v2_augmentation(img, spec)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)
        labels = [flip_label_x(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_x = IMG_SIZE - spec.bbox_x - spec.bbox_w
    if random.random() < 0.12:
        img = cv2.flip(img, 0)
        labels = [flip_label_y(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_y = IMG_SIZE - spec.bbox_y - spec.bbox_h

    return img, labels, spec


def apply_augmentation(img: np.ndarray, spec: SampleSpec) -> np.ndarray:
    side = max(1, spec.bbox_px_size)
    aug = spec.augmentation_type

    if aug == "clean":
        return img
    if aug == "basic_light":
        spec.sun_level = "mild"
        return color_light_jitter(img, hard=False)
    if aug == "light_blur":
        spec.blur_level = "light_defocus"
        return cv2.GaussianBlur(img, (3, 3), 0)
    if aug == "motion_blur":
        spec.blur_level = "motion_medium"
        max_len = int(np.clip(side * 0.45, 5, 17))
        choices = [k for k in [5, 7, 9, 11, 13, 15, 17] if k <= max_len] or [5]
        return apply_motion_blur(img, random.choice(choices), random.uniform(0, 180))
    if aug == "shadow":
        spec.shadow_level = "medium"
        return soft_shadow_light(img, "shadow")
    if aug == "sunny":
        spec.sun_level = "bright"
        return soft_shadow_light(img, "sunny")
    if aug == "partial_glare":
        spec.glare_level = "partial"
        return partial_glare(img, side, disaster=False)
    if aug == "blur_light":
        spec.blur_level = "motion_light"
        spec.sun_level = "mixed"
        max_len = int(np.clip(side * 0.50, 7, 19))
        choices = [k for k in [7, 9, 11, 13, 15, 17, 19] if k <= max_len] or [7]
        img = apply_motion_blur(img, random.choice(choices), random.uniform(0, 180))
        return soft_shadow_light(img, random.choice(["normal", "sunny", "shadow"]))
    if aug == "disaster":
        spec.blur_level = "disaster"
        spec.sun_level = "disaster"
        spec.glare_level = "disaster"
        spec.shadow_level = "hard"
        max_len = int(np.clip(side * 1.10, 11, 35))
        choices = [k for k in [11, 13, 15, 17, 19, 21, 25, 29, 33, 35] if k <= max_len] or [11]
        img = apply_motion_blur(img, random.choice(choices), random.uniform(0, 180))
        img = partial_glare(img, side, disaster=True)
        img = color_light_jitter(img, hard=True)
        img = add_noise(img, 10, 34)
        return jpeg_roundtrip(img, 28, 70)

    return img


def make_sample(backgrounds: list[tuple[Path, str]], stage: int) -> tuple[np.ndarray, list[str], SampleSpec]:
    img, bg_type = get_background(backgrounds, stage)
    target, aug = stage_recipe(stage)
    spec = SampleSpec(
        has_target=target in ("red", "blue"),
        target_color=target if target in ("red", "blue") else "none",
        target_type="square" if target in ("red", "blue") else ("negative" if target == "negative" else "background"),
        augmentation_type=aug,
        background_type=bg_type,
    )

    labels: list[str] = []
    if target in ("red", "blue"):
        img, label, bbox_size, bbox, angle, perspective = draw_target_square(img, stage, target)
        labels.append(label)
        x1, y1, x2, y2 = bbox
        spec.bbox_x = x1
        spec.bbox_y = y1
        spec.bbox_w = max(1, x2 - x1)
        spec.bbox_h = max(1, y2 - y1)
        spec.bbox_px_size = bbox_size
        spec.rotation_deg = angle
        spec.perspective_level = perspective
    elif target == "negative":
        n = 1 if random.random() < 0.82 else 2
        notes = []
        for _ in range(n):
            img, note = draw_negative_object(img, stage)
            notes.append(note)
        spec.notes = ";".join(notes)

    img = apply_stage1_augmentation(img, spec) if stage == 1 else apply_augmentation(img, spec)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)
        labels = [flip_label_x(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_x = IMG_SIZE - spec.bbox_x - spec.bbox_w
    if random.random() < 0.18:
        img = cv2.flip(img, 0)
        labels = [flip_label_y(lbl) for lbl in labels]
        if spec.has_target:
            spec.bbox_y = IMG_SIZE - spec.bbox_y - spec.bbox_h

    return img, labels, spec


def flip_label_x(lbl: str) -> str:
    parts = lbl.split()
    parts[1] = f"{1.0 - float(parts[1]):.6f}"
    return " ".join(parts)


def flip_label_y(lbl: str) -> str:
    parts = lbl.split()
    parts[2] = f"{1.0 - float(parts[2]):.6f}"
    return " ".join(parts)


def split_for_index(idx: int, total: int) -> str:
    r = idx / max(1, total)
    if r < 0.80:
        return "train"
    if r < 0.95:
        return "val"
    return "test"


def prepare_dataset_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Use --overwrite or choose another output.")
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir: Path) -> None:
    try:
        yaml_path_value = out_dir.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        yaml_path_value = str(out_dir.resolve())
    (out_dir / "data.yaml").write_text(
        f"""path: {yaml_path_value}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: target_square
""",
        encoding="utf-8",
    )


def make_preview(out_dir: Path, preview_items: list[tuple[np.ndarray, list[str], SampleSpec]], stage: int) -> None:
    if not preview_items:
        return
    thumbs = []
    for img, labels, spec in preview_items[:24]:
        vis = img.copy()
        if labels:
            x1, y1 = spec.bbox_x, spec.bbox_y
            x2, y2 = spec.bbox_x + spec.bbox_w, spec.bbox_y + spec.bbox_h
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        tag = spec.target_color if spec.has_target else spec.target_type
        if stage == 3:
            tag = f"{tag} {spec.blur_level}"
        if stage == 4:
            tag = f"{tag} {spec.augmentation_type[:8]}"
        cv2.putText(vis, tag, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(cv2.resize(vis, (160, 160), interpolation=cv2.INTER_AREA))
    cols = 6
    rows = math.ceil(len(thumbs) / cols)
    canvas = np.full((rows * 160, cols * 160, 3), 32, dtype=np.uint8)
    for i, img in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        canvas[y : y + 160, x : x + 160] = img
    cv2.imwrite(str(out_dir / "preview_grid.jpg"), canvas)


def sanity_check_stage1(out_dir: Path, expected_count: int) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "0: target_square" not in yaml_text:
        errors.append("data.yaml tek sinifli target_square degil.")

    image_paths = []
    label_paths = []
    for split in ("train", "val", "test"):
        imgs = sorted((out_dir / "images" / split).glob("*.jpg"))
        lbls = sorted((out_dir / "labels" / split).glob("*.txt"))
        image_paths.extend(imgs)
        label_paths.extend(lbls)
        if not imgs:
            warnings.append(f"{split} split icinde goruntu yok.")
        if len(imgs) != len(lbls):
            errors.append(f"{split} split image/label sayisi uyusmuyor: {len(imgs)} / {len(lbls)}")

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != len(image_paths):
        errors.append(f"metadata satir sayisi image sayisiyla uyusmuyor: {len(rows)} / {len(image_paths)}")
    if len(image_paths) != expected_count:
        errors.append(f"image sayisi beklenen count ile uyusmuyor: {len(image_paths)} / {expected_count}")

    row_by_name = {r["image_name"]: r for r in rows}
    allowed_colors = {"red", "blue", "none"}
    for row in rows:
        if row["target_color"] not in allowed_colors:
            errors.append(f"metadata target_color gecersiz: {row['image_name']} -> {row['target_color']}")

    for img_path in image_paths:
        split = img_path.parent.name
        lbl_path = out_dir / "labels" / split / f"{img_path.stem}.txt"
        row = row_by_name.get(img_path.name)
        if row is None:
            errors.append(f"metadata satiri yok: {img_path.name}")
            continue

        label_text = lbl_path.read_text(encoding="utf-8").strip()
        if row["has_target"] == "0":
            if label_text:
                errors.append(f"bos goruntunun label dosyasi bos degil: {lbl_path.name}")
        else:
            if not label_text:
                errors.append(f"hedefli goruntunun label dosyasi bos: {lbl_path.name}")

        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"label formati bozuk: {lbl_path.name}")
                continue
            if parts[0] != "0":
                errors.append(f"class id 0 degil: {lbl_path.name} -> {parts[0]}")
            vals = [float(v) for v in parts[1:]]
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"YOLO bbox 0-1 araliginda degil: {lbl_path.name}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            b, g, r = cv2.split(img)
            yellow = (b < 70) & (g > 225) & (r > 225)
            green = (b < 70) & (g > 225) & (r < 90)
            leak_ratio = float((yellow | green).sum()) / float(img.shape[0] * img.shape[1])
            if leak_ratio > 0.002:
                warnings.append(f"preview renklerine benzeyen sari/yesil iz olabilir: {img_path.name} ({leak_ratio:.4f})")

    if errors:
        raise RuntimeError("Stage 1 sanity check failed:\n- " + "\n- ".join(errors[:20]))

    print("Stage 1 sanity check OK")
    if warnings:
        print("Stage 1 sanity warnings:")
        for warning in warnings[:12]:
            print(f"  - {warning}")


def sanity_check_stage2_v2(out_dir: Path, expected_count: int, label: str = "Stage 2 v2") -> None:
    errors: list[str] = []
    warnings: list[str] = []

    yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "0: target_square" not in yaml_text:
        errors.append("data.yaml tek sinifli target_square degil.")

    image_paths = []
    for split in ("train", "val", "test"):
        imgs = sorted((out_dir / "images" / split).glob("*.jpg"))
        lbls = sorted((out_dir / "labels" / split).glob("*.txt"))
        image_paths.extend(imgs)
        if not imgs:
            warnings.append(f"{split} split icinde goruntu yok.")
        if len(imgs) != len(lbls):
            errors.append(f"{split} split image/label sayisi uyusmuyor: {len(imgs)} / {len(lbls)}")

    with (out_dir / "metadata.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames != STAGE2_METADATA_FIELDS:
            errors.append("metadata kolonlari Stage 2 v2 beklenen formatta degil.")

    if len(rows) != len(image_paths):
        errors.append(f"metadata satir sayisi image sayisiyla uyusmuyor: {len(rows)} / {len(image_paths)}")
    if len(image_paths) != expected_count:
        errors.append(f"image sayisi beklenen count ile uyusmuyor: {len(image_paths)} / {expected_count}")

    row_by_name = {r["image_name"]: r for r in rows}
    allowed_colors = {"red", "blue", "none"}
    target_counts = {"red": 0, "blue": 0, "background": 0, "negative": 0}
    size_counts = {"medium": 0, "large": 0, "small": 0, "very_small": 0}
    bg_counts: dict[str, int] = {}
    dark_count = 0

    for row in rows:
        if row["target_color"] not in allowed_colors:
            errors.append(f"metadata target_color gecersiz: {row['image_name']} -> {row['target_color']}")
        if row["target_type"] in target_counts:
            target_counts[row["target_type"]] += 1
        if row["target_color"] in ("red", "blue"):
            target_counts[row["target_color"]] += 1
        bg_counts[row["background_type"]] = bg_counts.get(row["background_type"], 0) + 1
        for bucket in size_counts:
            if f"size_bucket={bucket}" in row["notes"]:
                size_counts[bucket] += 1

    for img_path in image_paths:
        split = img_path.parent.name
        lbl_path = out_dir / "labels" / split / f"{img_path.stem}.txt"
        row = row_by_name.get(img_path.name)
        if row is None:
            errors.append(f"metadata satiri yok: {img_path.name}")
            continue

        label_text = lbl_path.read_text(encoding="utf-8").strip()
        if row["target_type"] in {"background", "negative"}:
            if label_text:
                errors.append(f"empty/negative label dosyasi bos degil: {lbl_path.name}")
        else:
            if not label_text:
                errors.append(f"hedefli goruntunun label dosyasi bos: {lbl_path.name}")

        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"label formati bozuk: {lbl_path.name}")
                continue
            if parts[0] != "0":
                errors.append(f"class id 0 degil: {lbl_path.name} -> {parts[0]}")
            vals = [float(v) for v in parts[1:]]
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"YOLO bbox 0-1 araliginda degil: {lbl_path.name}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < 38:
                dark_count += 1
            b, g, r = cv2.split(img)
            yellow = (b < 70) & (g > 225) & (r > 225)
            green = (b < 70) & (g > 225) & (r < 90)
            leak_ratio = float((yellow | green).sum()) / float(img.shape[0] * img.shape[1])
            if leak_ratio > 0.002:
                warnings.append(f"preview renklerine benzeyen sari/yesil iz olabilir: {img_path.name} ({leak_ratio:.4f})")

    total = max(1, len(rows))
    target_total = max(1, target_counts["red"] + target_counts["blue"])
    if dark_count / total > 0.05:
        errors.append(f"cok koyu arka plan orani yuksek: {dark_count}/{total}")
    if bg_counts.get("asphalt", 0) / total > 0.15:
        warnings.append(f"asfaltimsi arka plan orani yuksek olabilir: {bg_counts.get('asphalt', 0)}/{total}")

    ratios = {
        "medium": size_counts["medium"] / target_total,
        "large": size_counts["large"] / target_total,
        "small": size_counts["small"] / target_total,
        "very_small": size_counts["very_small"] / target_total,
    }
    if not (0.50 <= ratios["medium"] <= 0.60):
        warnings.append(f"orta boy hedef orani beklenen aralik disinda: {ratios['medium']:.2f}")
    if not (0.20 <= ratios["large"] <= 0.25):
        warnings.append(f"buyuk hedef orani beklenen aralik disinda: {ratios['large']:.2f}")
    if not (0.15 <= ratios["small"] <= 0.20):
        warnings.append(f"kucuk hedef orani beklenen aralik disinda: {ratios['small']:.2f}")
    if not (0.03 <= ratios["very_small"] <= 0.08):
        warnings.append(f"cok kucuk hedef orani beklenen aralik disinda: {ratios['very_small']:.2f}")

    if errors:
        raise RuntimeError(f"{label} sanity check failed:\n- " + "\n- ".join(errors[:20]))

    print(f"{label} sanity check OK")
    print(f"{label} target counts: red={target_counts['red']} blue={target_counts['blue']} empty={target_counts['background']} negative={target_counts['negative']}")
    print(f"{label} size buckets: {size_counts}")
    print(f"{label} backgrounds: {bg_counts}")
    if warnings:
        print(f"{label} sanity warnings:")
        for warning in warnings[:12]:
            print(f"  - {warning}")


def sanity_check_stage3_v2(out_dir: Path, expected_count: int) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "0: target_square" not in yaml_text:
        errors.append("data.yaml tek sinifli target_square degil.")

    image_paths = []
    for split in ("train", "val", "test"):
        imgs = sorted((out_dir / "images" / split).glob("*.jpg"))
        lbls = sorted((out_dir / "labels" / split).glob("*.txt"))
        image_paths.extend(imgs)
        if len(imgs) != len(lbls):
            errors.append(f"{split} split image/label sayisi uyusmuyor: {len(imgs)} / {len(lbls)}")

    with (out_dir / "metadata.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames != STAGE3_METADATA_FIELDS:
            errors.append("metadata kolonlari Stage 3 beklenen formatta degil.")

    if len(rows) != len(image_paths):
        errors.append(f"metadata satir sayisi image sayisiyla uyusmuyor: {len(rows)} / {len(image_paths)}")
    if len(image_paths) != expected_count:
        errors.append(f"image sayisi beklenen count ile uyusmuyor: {len(image_paths)} / {expected_count}")

    row_by_name = {r["image_name"]: r for r in rows}
    positive_blur_counts: dict[str, int] = {
        "clean": 0,
        "light_motion": 0,
        "medium_motion": 0,
        "light_defocus": 0,
    }
    target_counts = {"red": 0, "blue": 0, "background": 0, "negative": 0}
    bg_counts: dict[str, int] = {}
    dark_count = 0

    for row in rows:
        target_type = row["target_type"]
        if row["target_color"] in {"red", "blue"}:
            target_counts[row["target_color"]] += 1
        elif target_type in target_counts:
            target_counts[target_type] += 1
        bg_counts[row["background_type"]] = bg_counts.get(row["background_type"], 0) + 1

        if target_type == "square":
            blur_type = row["blur_type"]
            blur_level = row["blur_level"]
            notes = row["notes"]
            length = int(float(row["motion_blur_length"] or 0))
            if blur_type == "none":
                positive_blur_counts["clean"] += 1
            elif blur_type == "defocus":
                positive_blur_counts["light_defocus"] += 1
                if blur_level != "light":
                    errors.append(f"defocus sadece light olmali: {row['image_name']}")
            elif blur_type == "motion" and blur_level == "light":
                positive_blur_counts["light_motion"] += 1
            elif blur_type == "motion" and blur_level == "medium":
                positive_blur_counts["medium_motion"] += 1
            else:
                errors.append(f"gecersiz blur metadata: {row['image_name']} {blur_type}/{blur_level}")

            if "size_bucket=very_small" in notes and blur_type == "motion" and length > 5:
                errors.append(f"very_small hedefte blur length fazla: {row['image_name']} length={length}")
            if "size_bucket=small" in notes and blur_type == "motion" and (blur_level == "medium" or length > 7):
                errors.append(f"small hedefte orta/fazla blur var: {row['image_name']} level={blur_level} length={length}")
            if "size_bucket=very_small" in notes and blur_level == "medium":
                errors.append(f"very_small hedefte medium blur var: {row['image_name']}")

    for img_path in image_paths:
        split = img_path.parent.name
        lbl_path = out_dir / "labels" / split / f"{img_path.stem}.txt"
        row = row_by_name.get(img_path.name)
        if row is None:
            errors.append(f"metadata satiri yok: {img_path.name}")
            continue

        label_text = lbl_path.read_text(encoding="utf-8").strip()
        if row["target_type"] in {"background", "negative"}:
            if label_text:
                errors.append(f"empty/negative label dosyasi bos degil: {lbl_path.name}")
        else:
            if not label_text:
                errors.append(f"hedefli goruntunun label dosyasi bos: {lbl_path.name}")

        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"label formati bozuk: {lbl_path.name}")
                continue
            if parts[0] != "0":
                errors.append(f"class id 0 degil: {lbl_path.name} -> {parts[0]}")
            vals = [float(v) for v in parts[1:]]
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"YOLO bbox 0-1 araliginda degil: {lbl_path.name}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < 38:
                dark_count += 1
            b, g, r = cv2.split(img)
            yellow = (b < 70) & (g > 225) & (r > 225)
            green = (b < 70) & (g > 225) & (r < 90)
            leak_ratio = float((yellow | green).sum()) / float(img.shape[0] * img.shape[1])
            if leak_ratio > 0.002:
                warnings.append(f"preview renklerine benzeyen sari/yesil iz olabilir: {img_path.name} ({leak_ratio:.4f})")

    total = max(1, len(rows))
    positive_total = max(1, sum(positive_blur_counts.values()))
    expected = {
        "clean": (0.40, 0.55),
        "light_motion": (0.22, 0.32),
        "medium_motion": (0.12, 0.22),
        "light_defocus": (0.03, 0.08),
    }
    for key, (lo, hi) in expected.items():
        ratio = positive_blur_counts[key] / positive_total
        if not (lo <= ratio <= hi):
            warnings.append(f"{key} pozitif orani beklenen aralik disinda: {ratio:.2f}")

    if dark_count / total > 0.05:
        errors.append(f"cok koyu arka plan orani yuksek: {dark_count}/{total}")
    if target_counts["red"] != target_counts["blue"]:
        warnings.append(f"red/blue pozitif dengesi kaymis: {target_counts['red']}/{target_counts['blue']}")

    if errors:
        raise RuntimeError("Stage 3 v2 sanity check failed:\n- " + "\n- ".join(errors[:24]))

    print("Stage 3 clean-motion sanity check OK")
    print(f"Stage 3 target counts: red={target_counts['red']} blue={target_counts['blue']} empty={target_counts['background']} negative={target_counts['negative']}")
    print(f"Stage 3 positive blur counts: {positive_blur_counts}")
    print(f"Stage 3 backgrounds: {bg_counts}")
    if warnings:
        print("Stage 3 sanity warnings:")
        for warning in warnings[:12]:
            print(f"  - {warning}")


def sanity_check_stage4_v2(out_dir: Path, expected_count: int) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    if "nc: 1" not in yaml_text or "0: target_square" not in yaml_text:
        errors.append("data.yaml tek sinifli target_square degil.")

    image_paths = []
    for split in ("train", "val", "test"):
        imgs = sorted((out_dir / "images" / split).glob("*.jpg"))
        lbls = sorted((out_dir / "labels" / split).glob("*.txt"))
        image_paths.extend(imgs)
        if len(imgs) != len(lbls):
            errors.append(f"{split} split image/label sayisi uyusmuyor: {len(imgs)} / {len(lbls)}")

    with (out_dir / "metadata.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames != STAGE4_METADATA_FIELDS:
            errors.append("metadata kolonlari Stage 4 beklenen formatta degil.")

    if len(rows) != len(image_paths):
        errors.append(f"metadata satir sayisi image sayisiyla uyusmuyor: {len(rows)} / {len(image_paths)}")
    if len(image_paths) != expected_count:
        errors.append(f"image sayisi beklenen count ile uyusmuyor: {len(image_paths)} / {expected_count}")

    row_by_name = {r["image_name"]: r for r in rows}
    target_counts = {"red": 0, "blue": 0, "background": 0, "negative": 0}
    aug_counts: dict[str, int] = {}
    positive_aug_counts: dict[str, int] = {}
    bg_counts: dict[str, int] = {}
    dark_count = 0
    bright_count = 0
    color_shift_count = 0

    for row in rows:
        if row["target_color"] in {"red", "blue"}:
            target_counts[row["target_color"]] += 1
        elif row["target_type"] in target_counts:
            target_counts[row["target_type"]] += 1
        aug = row["notes"].split("augmentation=")[-1].split(";")[0] if "augmentation=" in row["notes"] else "unknown"
        aug_counts[aug] = aug_counts.get(aug, 0) + 1
        bg_counts[row["background_type"]] = bg_counts.get(row["background_type"], 0) + 1
        if row["color_shift_level"] != "none":
            color_shift_count += 1
        if row["target_type"] == "square":
            positive_aug_counts[aug] = positive_aug_counts.get(aug, 0) + 1
            notes = row["notes"]
            if ("size_bucket=very_small" in notes or "size_bucket=small" in notes) and row["lighting_type"] in {"glare", "blur_light"}:
                errors.append(f"kucuk hedefe agir Stage 4 efekti uygulanmis: {row['image_name']} lighting={row['lighting_type']}")
            if "size_bucket=very_small" in notes and row["shadow_level"] == "medium":
                errors.append(f"very_small hedefte medium shadow var: {row['image_name']}")
            if row["glare_level"] not in {"none", "light", "medium"}:
                errors.append(f"gecersiz glare_level: {row['image_name']} -> {row['glare_level']}")
            if row["color_shift_level"] not in {"none", "light"}:
                errors.append(f"asiri color shift: {row['image_name']} -> {row['color_shift_level']}")

    for img_path in image_paths:
        split = img_path.parent.name
        lbl_path = out_dir / "labels" / split / f"{img_path.stem}.txt"
        row = row_by_name.get(img_path.name)
        if row is None:
            errors.append(f"metadata satiri yok: {img_path.name}")
            continue

        label_text = lbl_path.read_text(encoding="utf-8").strip()
        if row["target_type"] in {"background", "negative"}:
            if label_text:
                errors.append(f"empty/negative label dosyasi bos degil: {lbl_path.name}")
        else:
            if not label_text:
                errors.append(f"hedefli goruntunun label dosyasi bos: {lbl_path.name}")

        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"label formati bozuk: {lbl_path.name}")
                continue
            if parts[0] != "0":
                errors.append(f"class id 0 degil: {lbl_path.name} -> {parts[0]}")
            vals = [float(v) for v in parts[1:]]
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"YOLO bbox 0-1 araliginda degil: {lbl_path.name}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_v = float(gray.mean())
            if mean_v < 30:
                dark_count += 1
            if mean_v > 235:
                bright_count += 1
            b, g, r = cv2.split(img)
            yellow = (b < 70) & (g > 225) & (r > 225)
            green = (b < 70) & (g > 225) & (r < 90)
            leak_ratio = float((yellow | green).sum()) / float(img.shape[0] * img.shape[1])
            if leak_ratio > 0.002:
                warnings.append(f"preview renklerine benzeyen sari/yesil iz olabilir: {img_path.name} ({leak_ratio:.4f})")

    total = max(1, len(rows))
    if dark_count / total > 0.06:
        errors.append(f"asiri karanlik goruntu orani yuksek: {dark_count}/{total}")
    if bright_count / total > 0.06:
        errors.append(f"asiri parlak goruntu orani yuksek: {bright_count}/{total}")
    if color_shift_count / total > 0.35:
        errors.append(f"color shift orani fazla: {color_shift_count}/{total}")
    positive_total = max(1, sum(positive_aug_counts.values()))
    expected_positive = {
        "clean": (0.28, 0.39),
        "shadow": (0.18, 0.26),
        "sunny": (0.18, 0.26),
        "partial_glare": (0.08, 0.14),
        "blur_light": (0.08, 0.14),
    }
    for key, (lo, hi) in expected_positive.items():
        ratio = positive_aug_counts.get(key, 0) / positive_total
        if not (lo <= ratio <= hi):
            warnings.append(f"{key} pozitif orani beklenen aralik disinda: {ratio:.2f}")
    if target_counts["red"] != target_counts["blue"]:
        warnings.append(f"red/blue pozitif dengesi kaymis: {target_counts['red']}/{target_counts['blue']}")

    if errors:
        raise RuntimeError("Stage 4 v2 sanity check failed:\n- " + "\n- ".join(errors[:24]))

    print("Stage 4 controlled sun-shadow sanity check OK")
    print(f"Stage 4 target counts: red={target_counts['red']} blue={target_counts['blue']} empty={target_counts['background']} negative={target_counts['negative']}")
    print(f"Stage 4 augmentation counts: {aug_counts}")
    print(f"Stage 4 positive augmentation counts: {positive_aug_counts}")
    print(f"Stage 4 backgrounds: {bg_counts}")
    if warnings:
        print("Stage 4 sanity warnings:")
        for warning in warnings[:12]:
            print(f"  - {warning}")


def generate_stage(
    stage: int,
    count: int,
    backgrounds: list[tuple[Path, str]],
    out_root: Path,
    overwrite: bool,
    variant: str = "default",
) -> Path:
    if stage == 2 and variant == "v2_preview":
        out_dir = out_root / STAGE2_V2_DIR
    elif stage == 3 and variant == "v2_preview":
        out_dir = out_root / STAGE3_V2_DIR
    elif stage == 4 and variant == "v2_preview":
        out_dir = out_root / STAGE4_V2_DIR
    else:
        out_dir = out_root / STAGE_DIRS[stage]
    prepare_dataset_dir(out_dir, overwrite=overwrite)

    indices = list(range(count))
    random.shuffle(indices)
    split_map = {idx: split_for_index(pos, count) for pos, idx in enumerate(indices)}
    rows = []
    preview_items: list[tuple[np.ndarray, list[str], SampleSpec]] = []
    stage1_plan = build_stage1_plan(count) if stage == 1 else []
    stage2_v2_plan = build_stage2_v2_plan(count) if stage == 2 and variant == "v2_preview" else []
    stage3_v2_plan = build_stage3_v2_plan(count) if stage == 3 and variant == "v2_preview" else []
    stage4_v2_plan = build_stage4_v2_plan(count) if stage == 4 and variant == "v2_preview" else []

    for idx in tqdm(range(count), desc=f"Stage {stage}", unit="img"):
        split = split_map[idx]
        if stage == 1:
            img, labels, spec = make_stage1_sample(stage1_plan[idx])
        elif stage == 2 and variant == "v2_preview":
            img, labels, spec = make_stage2_v2_sample(backgrounds, stage2_v2_plan[idx])
        elif stage == 3 and variant == "v2_preview":
            img, labels, spec = make_stage3_v2_sample(backgrounds, stage3_v2_plan[idx])
        elif stage == 4 and variant == "v2_preview":
            img, labels, spec = make_stage4_v2_sample(backgrounds, stage4_v2_plan[idx])
        else:
            img, labels, spec = make_sample(backgrounds, stage)
        prefix = f"stage{stage}v2" if variant == "v2_preview" else f"stage{stage}"
        name = f"{prefix}_{idx:06d}.jpg"
        label_name = f"{prefix}_{idx:06d}.txt"

        cv2.imwrite(str(out_dir / "images" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        (out_dir / "labels" / split / label_name).write_text("\n".join(labels), encoding="utf-8")

        if len(preview_items) < 24:
            preview_items.append((img.copy(), labels.copy(), spec))

        rows.append({
            "image_name": name,
            "split": split,
            "stage": STAGE_NAMES[stage],
            "has_target": int(spec.has_target),
            "target_color": spec.target_color,
            "target_type": spec.target_type,
            "background_type": spec.background_type,
            "negative_type": spec.negative_type,
            "augmentation_type": spec.augmentation_type,
            "bbox_x": spec.bbox_x,
            "bbox_y": spec.bbox_y,
            "bbox_w": spec.bbox_w,
            "bbox_h": spec.bbox_h,
            "bbox_px_size": spec.bbox_px_size,
            "rotation_deg": f"{spec.rotation_deg:.3f}",
            "perspective_level": f"{spec.perspective_level:.5f}",
            "blur_type": spec.blur_type,
            "blur_level": spec.blur_level,
            "motion_blur_length": spec.motion_blur_length,
            "motion_blur_angle": f"{spec.motion_blur_angle:.3f}",
            "sun_level": spec.sun_level,
            "glare_level": spec.glare_level,
            "shadow_level": spec.shadow_level,
            "lighting_type": spec.lighting_type,
            "lighting_level": spec.lighting_level,
            "exposure_level": spec.exposure_level,
            "color_shift_level": spec.color_shift_level,
            "disaster_level": spec.disaster_level,
            "notes": spec.notes,
        })

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        if stage == 2 and variant == "v2_preview":
            fieldnames = STAGE2_METADATA_FIELDS
        elif stage == 3 and variant == "v2_preview":
            fieldnames = STAGE3_METADATA_FIELDS
        elif stage == 4 and variant == "v2_preview":
            fieldnames = STAGE4_METADATA_FIELDS
        else:
            fieldnames = METADATA_FIELDS
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_data_yaml(out_dir)
    make_preview(out_dir, preview_items, stage)
    if stage == 1:
        sanity_check_stage1(out_dir, count)
    if stage == 2 and variant == "v2_preview":
        sanity_check_stage2_v2(out_dir, count)
    if stage == 3 and variant == "v2_preview":
        sanity_check_stage3_v2(out_dir, count)
    if stage == 4 and variant == "v2_preview":
        sanity_check_stage4_v2(out_dir, count)
    return out_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ASFLY single-class curriculum dataset generator")
    ap.add_argument("--bg-dir", default=str(DEFAULT_BG_DIR), help="Background image directory")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root for stage datasets")
    ap.add_argument("--stage", default="all", choices=["1", "2", "2v2", "3", "3v2", "4", "4v2", "all"], help="Stage to generate")
    ap.add_argument("--count", type=int, default=4000, help="Images per selected stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="Delete and recreate selected output folders")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_dir = Path(args.bg_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    backgrounds = load_backgrounds(bg_dir)

    if args.stage == "all":
        stages = [(1, "default"), (2, "default"), (3, "default"), (4, "default")]
    elif args.stage == "2v2":
        stages = [(2, "v2_preview")]
    elif args.stage == "3v2":
        stages = [(3, "v2_preview")]
    elif args.stage == "4v2":
        stages = [(4, "v2_preview")]
    else:
        stages = [(int(args.stage), "default")]
    print(f"Backgrounds: {len(backgrounds)} from {bg_dir}")
    print("YOLO class: 0 target_square")
    for stage, variant in stages:
        out_dir = generate_stage(stage, args.count, backgrounds, out_root, args.overwrite, variant=variant)
        suffix = " v2 preview" if variant == "v2_preview" else ""
        print(f"Stage {stage}{suffix} ready: {out_dir}")


if __name__ == "__main__":
    main()
