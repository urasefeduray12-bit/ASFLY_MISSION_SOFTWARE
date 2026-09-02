from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentController:
    image_width: int
    image_height: int
    center_tolerance_px: float
    max_speed_mps: float

    def is_centered(self, center: tuple[float, float]) -> bool:
        error_x = center[0] - self.image_width / 2.0
        error_y = center[1] - self.image_height / 2.0
        return abs(error_x) <= self.center_tolerance_px and abs(error_y) <= self.center_tolerance_px

    def velocity_from_center(self, center: tuple[float, float], forward_sign: float = 1.0) -> tuple[float, float]:
        """Return XY velocity in the mission navigation frame."""
        error_x = center[0] - self.image_width / 2.0
        error_y = center[1] - self.image_height / 2.0
        norm_x = max(-1.0, min(1.0, error_x / max(1.0, self.image_width / 2.0)))
        norm_y = max(-1.0, min(1.0, error_y / max(1.0, self.image_height / 2.0)))
        direction = 1.0 if forward_sign >= 0.0 else -1.0
        
        # Kamera ve dron yönü 180 derece ters (Kuzey-Güney) olduğu için
        # eksi işaretleri kaldırıp eksen komutlarını tersine çevirdik.
        vx = norm_y * self.max_speed_mps * direction
        vy = norm_x * self.max_speed_mps * direction
        return vx, vy