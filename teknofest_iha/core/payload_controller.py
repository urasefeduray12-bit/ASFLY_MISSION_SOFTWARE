from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PayloadController:
    released_targets: set[str]

    def __init__(self) -> None:
        self.released_targets = set()

    def can_release(self, target_type: str) -> bool:
        return target_type not in self.released_targets

    def mark_released(self, target_type: str) -> None:
        self.released_targets.add(target_type)
