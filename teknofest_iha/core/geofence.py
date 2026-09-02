from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GeofenceLevel(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"


@dataclass(frozen=True)
class Geofence:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    warning_margin_m: float = 2.0
    hard_margin_m: float = 0.5

    def check(self, x: float, y: float) -> GeofenceLevel:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return GeofenceLevel.VIOLATION
        if (
            x <= self.x_min + self.warning_margin_m
            or x >= self.x_max - self.warning_margin_m
            or y <= self.y_min + self.warning_margin_m
            or y >= self.y_max - self.warning_margin_m
        ):
            return GeofenceLevel.WARNING
        return GeofenceLevel.OK

    def clamp_velocity(self, x: float, y: float, vx: float, vy: float) -> tuple[float, float]:
        next_vx = vx
        next_vy = vy
        if x <= self.x_min + self.hard_margin_m and vx < 0:
            next_vx = 0.0
        if x >= self.x_max - self.hard_margin_m and vx > 0:
            next_vx = 0.0
        if y <= self.y_min + self.hard_margin_m and vy < 0:
            next_vy = 0.0
        if y >= self.y_max - self.hard_margin_m and vy > 0:
            next_vy = 0.0
        return next_vx, next_vy
