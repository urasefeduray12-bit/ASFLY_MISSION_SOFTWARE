#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SCAN_X_MIN = 0.0
SCAN_X_MAX = 100.0
SCAN_Y_MIN = -15.0
SCAN_Y_MAX = 15.0


@dataclass(frozen=True)
class ObjectSpec:
    model_name: str
    metric_name: str
    radius_m: float
    pose_z: float
    size: tuple[float, float] | None = None


@dataclass(frozen=True)
class Placement:
    spec: ObjectSpec
    x: float
    y: float


RANDOM_TARGETS = (
    ObjectSpec("red_square_target", "red_square", 0.75, 0.04, (1.0, 1.0)),
    ObjectSpec("blue_square_target", "blue_square", 1.35, 0.04, (2.0, 2.0)),
)

FIXED_DISTRACTORS = (
    Placement(ObjectSpec("no_drop_circle_2m", "no_drop_circle_2m", 1.0, 0.035), 30.0, 7.0),
    Placement(ObjectSpec("no_drop_circle_10m", "no_drop_circle_10m", 5.0, 0.035), 60.0, 7.0),
)


def generate_placements(seed: int | None) -> list[Placement]:
    rng = random.Random(seed)
    placements: list[Placement] = []
    for spec in RANDOM_TARGETS:
        margin = spec.radius_m + 2.0
        for _ in range(10_000):
            x = rng.uniform(SCAN_X_MIN + margin, SCAN_X_MAX - margin)
            y = rng.uniform(SCAN_Y_MIN + margin, SCAN_Y_MAX - margin)
            candidate = Placement(spec, round(x, 2), round(y, 2))
            if all(_clear(candidate, other) for other in [*placements, *FIXED_DISTRACTORS]):
                placements.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not place {spec.model_name} without overlap")
    return placements


def _clear(candidate: Placement, other: Placement) -> bool:
    distance = math.hypot(candidate.x - other.x, candidate.y - other.y)
    return distance >= candidate.spec.radius_m + other.spec.radius_m + 4.0


def update_world(world_path: Path, placements: list[Placement]) -> None:
    tree = ET.parse(world_path)
    root = tree.getroot()
    for placement in [*placements, *FIXED_DISTRACTORS]:
        model = _find_model(root, placement.spec.model_name)
        pose = model.find("pose")
        if pose is None:
            raise RuntimeError(f"Model {placement.spec.model_name} has no pose")
        pose.text = f"{placement.x:.2f} {placement.y:.2f} {placement.spec.pose_z:.3f} 0 0 0"
    ET.indent(tree, space="  ")
    tree.write(world_path, encoding="utf-8", xml_declaration=True)


def _find_model(root: ET.Element, name: str) -> ET.Element:
    for model in root.findall(".//model"):
        if model.attrib.get("name") == name:
            return model
    raise RuntimeError(f"Model not found in SDF: {name}")


def update_mission_config(config_path: Path, placements: list[Placement]) -> None:
    target_specs = {}
    for placement in placements:
        if placement.spec.size is None:
            continue
        target_specs[placement.spec.metric_name] = {
            "center": [placement.x, placement.y],
            "size": list(placement.spec.size),
        }
    replacement = "target_specs_json: '" + json.dumps(target_specs, separators=(",", ":")) + "'"
    text = config_path.read_text(encoding="utf-8")
    new_text, count = re.subn(r"^(\s*)target_specs_json:\s*'.*'$", r"\1" + replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update target_specs_json in {config_path}")
    config_path.write_text(new_text, encoding="utf-8")


def backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_random_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def summarize(placements: list[Placement]) -> dict[str, dict[str, float]]:
    return {
        placement.spec.model_name: {"x": placement.x, "y": placement.y, "radius_m": placement.spec.radius_m}
        for placement in placements
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomize square target positions inside the 100x30 m scan area.")
    parser.add_argument("--world", default="worlds/teknofest_bozkir.sdf", type=Path)
    parser.add_argument("--mission", default="config/mission.yaml", type=Path)
    parser.add_argument("--external-world", action="append", default=[], type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    placements = generate_placements(args.seed)
    paths = [args.world, *args.external_world]

    if not args.dry_run:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            if not args.no_backup:
                backup(path)
            update_world(path, placements)
        update_mission_config(args.mission, placements)

    print(
        json.dumps(
            {
                "seed": args.seed,
                "random_targets": summarize(placements),
                "fixed_distractors": summarize(list(FIXED_DISTRACTORS)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
