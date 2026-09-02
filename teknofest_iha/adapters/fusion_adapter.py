from __future__ import annotations

"""Adapter between raw detection packets and fusion target packets.

This module is where legacy OpenCV/YOLO fusion functions are made compatible
with the ROS-facing data model. It also owns one target fusion state machine per
mission target. It does not release payloads; it only produces release_gate.
"""

import time
from typing import Any

import config
from control.state_machine import TargetStateMachine, yolo_interval_for_state
from control.payload_logic import compute_release_gate
from vision.fusion import fuse_detections, yolo_result_is_fresh

from teknofest_iha.interfaces.detection_models import FusedTargetPacket, RawDetectionPacket


class FusionAdapter:
    """Fuses per-target OpenCV detections with YOLO shape verification."""

    def __init__(self, primary_target: str, secondary_targets: list[str] | None = None) -> None:
        self.primary_target = primary_target
        self.targets = [primary_target, *(secondary_targets or [])]
        self.state_machine_by_target = {target: TargetStateMachine() for target in self.targets}

    def fuse_packet(self, packet: RawDetectionPacket) -> FusedTargetPacket:
        fused_targets: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        selected_state = "SEARCH"
        yolo_fresh = self._packet_has_fresh_yolo(packet)

        for target in self.targets:
            opencv_dets = [d for d in packet.opencv if d.get("target_type") == target]
            yolo_dets = [
                d
                for d in (packet.yolo if yolo_fresh else [])
                if d.get("target_type") in (target, "square", "square_unknown")
            ]
            fused = fuse_detections(opencv_dets, yolo_dets, packet.frame_id)
            state_machine = self.state_machine_by_target[target]
            state = state_machine.update(
                fused,
                has_opencv_target=bool(opencv_dets),
                has_recent_yolo=yolo_fresh,
                frame_id=packet.frame_id,
            )
            release_gate = compute_release_gate(fused, state_machine.lock_counter)
            if fused is not None:
                fused = dict(fused)
                fused["target_state"] = state
                fused["release_gate"] = bool(release_gate)
                fused["drop_ready"] = bool(release_gate)
                fused["yolo_fresh"] = bool(yolo_fresh)
                fused["yolo_age_frames"] = packet.yolo_age_frames
                fused["lock_counter"] = int(state_machine.lock_counter)
                fused["unstable_counter"] = int(state_machine.unstable_counter)
                fused_targets.append(fused)
                if target == self.primary_target:
                    selected = fused
                    selected_state = state

        if selected is None and fused_targets:
            selected = fused_targets[0]
            selected_state = str(selected.get("target_state", "CANDIDATE"))

        return FusedTargetPacket(
            frame_id=packet.frame_id,
            timestamp=time.time(),
            primary_target=self.primary_target,
            state=selected_state,
            targets=fused_targets,
            selected=selected,
        )

    @staticmethod
    def yolo_is_fresh(yolo_result: dict | None, frame_id: int) -> bool:
        return yolo_result_is_fresh(yolo_result, frame_id)

    @staticmethod
    def yolo_interval(state: str) -> int:
        return yolo_interval_for_state(state)

    @staticmethod
    def _packet_has_fresh_yolo(packet: RawDetectionPacket) -> bool:
        if not packet.yolo_ran:
            return False
        if packet.yolo_age_frames is None:
            return False
        return 0 <= int(packet.yolo_age_frames) <= config.MAX_YOLO_RESULT_AGE_FRAMES
