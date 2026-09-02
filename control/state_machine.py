"""Target-level fusion state machine.

This state machine answers only one question: how reliable is the current
target observation? It does not know about takeoff, RTL, servo channels, or
payload execution. Payload permission is expressed separately as release_gate.
"""

import config


SEARCH = "SEARCH"
CANDIDATE = "CANDIDATE"
TRACKING = "TRACKING"
UNSTABLE = "UNSTABLE"
LOCKED = "LOCKED"


class TargetStateMachine:
    def __init__(self):
        self.state = SEARCH
        self.cv_seen_counter = 0
        self.lock_counter = 0
        self.unstable_counter = 0
        self.last_frame_id = None
        self.payload_released = False

    def update(self, fused_target, has_opencv_target, has_recent_yolo, frame_id=None):
        if frame_id is not None and frame_id == self.last_frame_id:
            return self.state
        self.last_frame_id = frame_id

        if not has_opencv_target or fused_target is None:
            self.cv_seen_counter = 0
            self.lock_counter = 0
            self.unstable_counter = 0
            self.state = SEARCH
            return self.state

        self.cv_seen_counter += 1
        verified = bool(fused_target.get("yolo_verified", False))
        fusion_conf = float(fused_target.get("fusion_confidence", 0.0))
        lock_condition = verified and fusion_conf >= config.FUSION_CONF_THRESH

        if self.cv_seen_counter <= config.CANDIDATE_MIN_FRAMES:
            self.lock_counter = 0
            self.unstable_counter = 0
            self.state = CANDIDATE
        elif lock_condition:
            self.unstable_counter = 0
            self.lock_counter += 1
            self.state = LOCKED
        elif has_recent_yolo and not verified:
            self.lock_counter = 0
            self.unstable_counter += 1
            if self.unstable_counter >= config.UNSTABLE_MIN_FRAMES:
                self.state = UNSTABLE
            else:
                self.state = TRACKING
        else:
            self.lock_counter = 0
            self.unstable_counter = 0
            self.state = TRACKING

        return self.state

    def mark_payload_released(self):
        self.payload_released = True


def yolo_interval_for_state(state):
    return {
        SEARCH: config.YOLO_EVERY_SEARCH,
        CANDIDATE: config.YOLO_EVERY_CANDIDATE,
        TRACKING: config.YOLO_EVERY_TRACKING,
        UNSTABLE: config.YOLO_EVERY_UNSTABLE,
        LOCKED: config.YOLO_EVERY_LOCKED,
    }.get(state, config.YOLO_EVERY_SEARCH)
