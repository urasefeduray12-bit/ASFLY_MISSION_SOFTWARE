from __future__ import annotations

from dataclasses import dataclass


BLUE_SQUARE = "blue_square"
RED_SQUARE = "red_square"
SQUARE_UNKNOWN = "square_unknown"


@dataclass(frozen=True)
class TargetPriority:
    primary: str = BLUE_SQUARE
    secondary: tuple[str, ...] = (RED_SQUARE,)

    def all_targets(self) -> tuple[str, ...]:
        return (self.primary, *self.secondary)
