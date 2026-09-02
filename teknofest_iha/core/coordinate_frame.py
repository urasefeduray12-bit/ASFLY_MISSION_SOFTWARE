from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateFrameMapper:
    """Maps MAVLink local NED XY into the mission navigation XY frame."""

    mode: str = "identity"

    def nav_xy_from_local(self, local_x: float, local_y: float) -> tuple[float, float]:
        if self.mode == "gazebo_xy_swapped":
            return local_y, local_x
        return local_x, local_y

    def local_velocity_from_nav(self, nav_vx: float, nav_vy: float) -> tuple[float, float]:
        if self.mode == "gazebo_xy_swapped":
            return nav_vy, nav_vx
        return nav_vx, nav_vy

    def nav_velocity_from_local(self, local_vx: float, local_vy: float) -> tuple[float, float]:
        if self.mode == "gazebo_xy_swapped":
            return local_vy, local_vx
        return local_vx, local_vy
