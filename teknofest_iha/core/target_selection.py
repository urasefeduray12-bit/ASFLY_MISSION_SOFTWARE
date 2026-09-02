from __future__ import annotations

from typing import Any, Callable


_STATE_PRIORITY = {
    "DROP_READY": 4,
    "LOCKED": 3,
    "TRACKING": 2,
    "CANDIDATE": 1,
}


def choose_visible_unreleased_target(
    selected: dict[str, Any] | None,
    targets: list[dict[str, Any]],
    allowed_targets: tuple[str, ...],
    can_release: Callable[[str], bool],
) -> str | None:
    """Choose the best visible target that still needs a payload."""
    allowed = set(allowed_targets)
    candidates: list[tuple[float, str]] = []

    def add_candidate(target: dict[str, Any], selected_bonus: float) -> None:
        target_type = str(target.get("target_type", ""))
        if target_type not in allowed or not can_release(target_type):
            return
        state = str(target.get("target_state", target.get("state", "")))
        confidence = float(target.get("fusion_confidence", target.get("confidence", 0.0)))
        release_bonus = 1.0 if bool(target.get("release_gate", target.get("drop_ready", False))) else 0.0
        score = selected_bonus + _STATE_PRIORITY.get(state, 0) + release_bonus + confidence
        candidates.append((score, target_type))

    if selected is not None:
        add_candidate(selected, selected_bonus=0.25)
    for target in targets:
        add_candidate(target, selected_bonus=0.0)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
