from __future__ import annotations

import math
from dataclasses import dataclass


GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class TargetSpec:
    center_x: float
    center_y: float
    size_x: float
    size_y: float


@dataclass(frozen=True)
class PayloadDropEstimate:
    release_x: float
    release_y: float
    release_altitude_m: float
    release_vx: float
    release_vy: float
    fall_time_s: float
    estimated_impact_x: float
    estimated_impact_y: float
    target_x: float
    target_y: float
    error_x: float
    error_y: float
    distance_to_center_m: float
    inside_target_footprint: bool

    def as_dict(self) -> dict[str, float | bool | dict[str, float]]:
        return {
            "release_nav": {
                "x": self.release_x,
                "y": self.release_y,
                "altitude_m": self.release_altitude_m,
            },
            "release_velocity_nav": {
                "vx": self.release_vx,
                "vy": self.release_vy,
            },
            "fall_time_s": self.fall_time_s,
            "estimated_impact_nav": {
                "x": self.estimated_impact_x,
                "y": self.estimated_impact_y,
            },
            "target_center_nav": {
                "x": self.target_x,
                "y": self.target_y,
            },
            "error_nav": {
                "x": self.error_x,
                "y": self.error_y,
            },
            "distance_to_center_m": self.distance_to_center_m,
            "inside_target_footprint": self.inside_target_footprint,
        }


def estimate_payload_drop(
    release_x: float,
    release_y: float,
    release_altitude_m: float,
    release_vx: float,
    release_vy: float,
    target: TargetSpec,
    gravity_mps2: float = GRAVITY_MPS2,
) -> PayloadDropEstimate:
    altitude = max(0.0, float(release_altitude_m))
    gravity = max(1e-6, float(gravity_mps2))
    fall_time_s = math.sqrt((2.0 * altitude) / gravity) if altitude > 0.0 else 0.0
    impact_x = float(release_x) + float(release_vx) * fall_time_s
    impact_y = float(release_y) + float(release_vy) * fall_time_s
    error_x = impact_x - target.center_x
    error_y = impact_y - target.center_y
    inside = abs(error_x) <= target.size_x / 2.0 and abs(error_y) <= target.size_y / 2.0
    return PayloadDropEstimate(
        release_x=float(release_x),
        release_y=float(release_y),
        release_altitude_m=altitude,
        release_vx=float(release_vx),
        release_vy=float(release_vy),
        fall_time_s=fall_time_s,
        estimated_impact_x=impact_x,
        estimated_impact_y=impact_y,
        target_x=target.center_x,
        target_y=target.center_y,
        error_x=error_x,
        error_y=error_y,
        distance_to_center_m=math.hypot(error_x, error_y),
        inside_target_footprint=inside,
    )
