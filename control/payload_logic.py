"""Release-gate logic for target fusion.

The functions in this file decide whether perception/fusion evidence is strong
enough to allow a payload release. They do not command a servo; mission manager
and MAVLink bridge handle execution.
"""

import config


def compute_release_gate(fused_target, lock_counter):
    if fused_target is None:
        return False
    err = fused_target.get("error")
    if err is None:
        return False
    err_x, err_y = err
    return (
        bool(fused_target.get("yolo_verified", False))
        and float(fused_target.get("fusion_confidence", 0.0)) >= config.FUSION_CONF_THRESH
        and abs(err_x) <= config.CENTER_TOL_X
        and abs(err_y) <= config.CENTER_TOL_Y
        and lock_counter >= config.LOCK_MIN_FRAMES
    )


def compute_drop_ready(fused_target, lock_counter, payload_released=False):
    return not payload_released and compute_release_gate(fused_target, lock_counter)


def release_payload_stub():
    print("[SAFE MODE] WOULD RELEASE PAYLOAD - real MAVLink disabled")
