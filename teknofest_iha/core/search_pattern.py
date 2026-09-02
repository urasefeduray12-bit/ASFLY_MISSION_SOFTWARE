from __future__ import annotations

"""Lawnmower search pattern generator.

The class generates a serpentine route over the rectangular scan area and can
start from the side nearest to the vehicle. This avoids wasting time flying to a
far lane start before the first useful scan.
"""

from dataclasses import dataclass


@dataclass
class LawnmowerSearchPattern:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    lane_spacing_m: float

    def waypoints(self) -> list[tuple[float, float]]:
        return self._waypoints(start_from_x_max=False)

    def waypoints_from_nearest_start(self, x: float, y: float) -> list[tuple[float, float]]:
        return self._waypoints(start_from_x_max=self.start_from_x_max_is_nearest(x, y))

    def waypoints_from_start(self, start_from_x_max: bool) -> list[tuple[float, float]]:
        return self._waypoints(start_from_x_max=start_from_x_max)

    def start_from_x_max_is_nearest(self, x: float, y: float) -> bool:
        return self._distance_sq(x, y, self.x_max, self.y_min) < self._distance_sq(x, y, self.x_min, self.y_min)

    def next_velocity_from_nearest_start(
        self,
        x: float,
        y: float,
        waypoint_index: int,
        speed_mps: float,
        acceptance_radius_m: float = 1.0,
    ) -> tuple[int, float, float]:
        points = self.waypoints_from_nearest_start(x, y)
        return self._next_velocity_for_points(points, x, y, waypoint_index, speed_mps, acceptance_radius_m)

    def next_velocity_from_start(
        self,
        x: float,
        y: float,
        waypoint_index: int,
        speed_mps: float,
        acceptance_radius_m: float,
        start_from_x_max: bool,
    ) -> tuple[int, float, float]:
        points = self.waypoints_from_start(start_from_x_max)
        return self._next_velocity_for_points(points, x, y, waypoint_index, speed_mps, acceptance_radius_m)

    def _waypoints(self, start_from_x_max: bool) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        y = self.y_min
        lane = 0
        while y <= self.y_max + 1e-6:
            left_to_right = lane % 2 == 0
            if start_from_x_max:
                left_to_right = not left_to_right
            if left_to_right:
                points.append((self.x_min, y))
                points.append((self.x_max, y))
            else:
                points.append((self.x_max, y))
                points.append((self.x_min, y))
            y += max(0.1, self.lane_spacing_m)
            lane += 1
        return points

    def next_velocity(
        self,
        x: float,
        y: float,
        waypoint_index: int,
        speed_mps: float,
        acceptance_radius_m: float = 1.0,
    ) -> tuple[int, float, float]:
        points = self.waypoints()
        return self._next_velocity_for_points(points, x, y, waypoint_index, speed_mps, acceptance_radius_m)

    def _next_velocity_for_points(
        self,
        points: list[tuple[float, float]],
        x: float,
        y: float,
        waypoint_index: int,
        speed_mps: float,
        acceptance_radius_m: float,
    ) -> tuple[int, float, float]:
        if not points:
            return waypoint_index, 0.0, 0.0
        index = min(max(waypoint_index, 0), len(points) - 1)
        while index < len(points) - 1 and self._reached_or_passed(points, index, x, y, acceptance_radius_m):
            index += 1
        tx, ty = points[index]
        if index > 0:
            px, py = points[index - 1]
            sx = tx - px
            sy = ty - py
            if abs(sx) >= abs(sy):
                return index, self._axis_velocity(tx - x, speed_mps), 0.0
            return index, 0.0, self._axis_velocity(ty - y, speed_mps)

        dx = tx - x
        dy = ty - y
        dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        scale = speed_mps / dist
        return index, dx * scale, dy * scale

    @staticmethod
    def _axis_velocity(error: float, speed_mps: float) -> float:
        if abs(error) <= 1e-6:
            return 0.0
        return speed_mps if error > 0.0 else -speed_mps

    @staticmethod
    def _reached_or_passed(
        points: list[tuple[float, float]],
        index: int,
        x: float,
        y: float,
        acceptance_radius_m: float,
    ) -> bool:
        tx, ty = points[index]
        dx = tx - x
        dy = ty - y
        if (dx * dx + dy * dy) ** 0.5 <= acceptance_radius_m:
            return True
        if index <= 0:
            return False
        px, py = points[index - 1]
        sx = tx - px
        sy = ty - py
        if abs(sx) >= abs(sy):
            if sx > 0.0:
                return x >= tx - acceptance_radius_m
            if sx < 0.0:
                return x <= tx + acceptance_radius_m
            return True
        if sy > 0.0:
            return y >= ty - acceptance_radius_m
        if sy < 0.0:
            return y <= ty + acceptance_radius_m
        return True

    @staticmethod
    def _distance_sq(x: float, y: float, tx: float, ty: float) -> float:
        dx = tx - x
        dy = ty - y
        return dx * dx + dy * dy
