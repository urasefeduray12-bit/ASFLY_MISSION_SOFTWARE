from __future__ import annotations

"""Pure mission state machine.

This module contains no ROS, MAVLink, OpenCV, or YOLO code. It only receives a
`MissionInputs` snapshot and returns the next `MissionState`. Keeping it pure is
what makes the most important mission transitions unit-testable.
"""

import time
from dataclasses import dataclass, field

from teknofest_iha.core.mission_states import MissionState


@dataclass
class MissionInputs:
    camera_ready: bool = False
    mavlink_connected: bool = False
    guided: bool = False
    armed: bool = False
    altitude_m: float = 0.0
    target: dict | None = None
    target_centered: bool = False
    target_locked: bool = False
    safety_level: str = "OK"
    drop_done: bool = False
    return_confirmed: bool = False


@dataclass
class MissionStateMachine:
    takeoff_altitude_m: float
    altitude_tolerance_m: float
    post_drop_hover_s: float
    target_sequence: tuple[str, ...] = ("blue_square",)
    restore_altitude_tolerance_m: float | None = None
    state: MissionState = MissionState.INIT
    entered_at: float = field(default_factory=time.time)
    active_target_index: int = 0

    @property
    def active_target(self) -> str | None:
        if self.active_target_index >= len(self.target_sequence):
            return None
        return self.target_sequence[self.active_target_index]

    def transition(self, state: MissionState) -> None:
        if state != self.state:
            self.state = state
            self.entered_at = time.time()

    def elapsed(self) -> float:
        return time.time() - self.entered_at

    def update(self, inputs: MissionInputs) -> MissionState:
        if inputs.safety_level == "VIOLATION":
            self.transition(MissionState.FAILSAFE)
            return self.state

        if self.state == MissionState.INIT:
            self.transition(MissionState.WAIT_FOR_CAMERA)
        elif self.state == MissionState.WAIT_FOR_CAMERA and inputs.camera_ready:
            self.transition(MissionState.CONNECT_MAVLINK)
        elif self.state == MissionState.CONNECT_MAVLINK and inputs.mavlink_connected:
            self.transition(MissionState.SET_GUIDED)
        elif self.state == MissionState.SET_GUIDED and inputs.guided:
            self.transition(MissionState.ARM)
        elif self.state == MissionState.ARM and inputs.armed:
            self.transition(MissionState.TAKEOFF)
        elif self.state == MissionState.TAKEOFF:
            reached = abs(inputs.altitude_m - self.takeoff_altitude_m) <= self.altitude_tolerance_m
            if reached:
                self.transition(MissionState.SEARCH_TARGET)
        elif self.state == MissionState.SEARCH_TARGET:
            if inputs.target is not None:
                self.transition(MissionState.TARGET_CANDIDATE)
        elif self.state == MissionState.TARGET_CANDIDATE:
            if inputs.target is None:
                self.transition(MissionState.SEARCH_TARGET)
            else:
                self.transition(MissionState.TARGET_ALIGN)
        elif self.state == MissionState.TARGET_ALIGN:
            if inputs.target is None:
                self.transition(MissionState.SEARCH_TARGET)
            elif inputs.target_centered:
                self.transition(MissionState.TARGET_VERIFY)
        elif self.state == MissionState.TARGET_VERIFY:
            if inputs.target is None:
                self.transition(MissionState.SEARCH_TARGET)
            elif inputs.target_centered and inputs.target_locked:
                self.transition(MissionState.DROP_TARGET)
        elif self.state == MissionState.DROP_TARGET and inputs.drop_done:
            self.transition(MissionState.POST_DROP_HOVER)
        elif self.state == MissionState.POST_DROP_HOVER and self.elapsed() >= self.post_drop_hover_s:
            restore_tolerance = self.restore_altitude_tolerance_m
            if restore_tolerance is None:
                restore_tolerance = self.altitude_tolerance_m
            restored_altitude = inputs.altitude_m >= self.takeoff_altitude_m - restore_tolerance
            if not restored_altitude:
                return self.state
            self.active_target_index += 1
            if self.active_target is None:
                self.transition(MissionState.RETURN_HOME)
            else:
                self.transition(MissionState.SEARCH_TARGET)
        elif self.state == MissionState.RETURN_HOME and inputs.return_confirmed:
            self.transition(MissionState.MISSION_COMPLETE)
        return self.state
