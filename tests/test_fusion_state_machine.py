import config
from control.state_machine import CANDIDATE, LOCKED, TRACKING, UNSTABLE, TargetStateMachine
from teknofest_iha.adapters.fusion_adapter import FusionAdapter
from teknofest_iha.interfaces.detection_models import RawDetectionPacket


def _fused(yolo_verified=True, confidence=0.9):
    return {
        "target_type": "red_square",
        "error": (0.0, 0.0),
        "yolo_verified": yolo_verified,
        "fusion_confidence": confidence,
    }


def test_one_frame_advances_state_machine_once():
    machine = TargetStateMachine()

    assert machine.update(_fused(), True, True, frame_id=10) == CANDIDATE
    assert machine.update(_fused(), True, True, frame_id=10) == CANDIDATE

    assert machine.cv_seen_counter == 1
    assert machine.lock_counter == 0


def test_candidate_warmup_precedes_lock():
    machine = TargetStateMachine()

    assert machine.update(_fused(), True, True, frame_id=1) == CANDIDATE
    assert machine.update(_fused(), True, True, frame_id=2) == CANDIDATE
    assert machine.update(_fused(), True, True, frame_id=3) == LOCKED


def test_negative_yolo_observation_can_make_target_unstable():
    machine = TargetStateMachine()

    assert machine.update(_fused(yolo_verified=False, confidence=0.6), True, True, frame_id=1) == CANDIDATE
    assert machine.update(_fused(yolo_verified=False, confidence=0.6), True, True, frame_id=2) == CANDIDATE
    assert machine.update(_fused(yolo_verified=False, confidence=0.6), True, True, frame_id=3) == TRACKING
    assert machine.update(_fused(yolo_verified=False, confidence=0.6), True, True, frame_id=4) == UNSTABLE


def test_fusion_adapter_uses_yolo_ran_even_when_yolo_detection_list_is_empty():
    adapter = FusionAdapter("red_square", [])
    cv_det = {
        "source": "opencv",
        "target_type": "red_square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.9,
        "state": "DETECTED",
        "frame_id": 1,
        "error": (0.0, 0.0),
    }

    states = []
    for frame_id in range(1, 5):
        packet = RawDetectionPacket(
            frame_id=frame_id,
            timestamp=0.0,
            opencv=[{**cv_det, "frame_id": frame_id}],
            yolo=[],
            yolo_ran=True,
            yolo_frame_id=frame_id,
            yolo_age_frames=0,
        )
        states.append(adapter.fuse_packet(packet).selected["target_state"])

    assert states[-1] == UNSTABLE


def test_fusion_adapter_ignores_stale_yolo_detections():
    adapter = FusionAdapter("red_square", [])
    cv_det = {
        "source": "opencv",
        "target_type": "red_square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.95,
        "state": "DETECTED",
        "frame_id": 30,
        "error": (0.0, 0.0),
    }
    yolo_det = {
        "source": "yolo",
        "target_type": "square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.95,
        "state": "DETECTED",
        "frame_id": 1,
    }
    packet = RawDetectionPacket(
        frame_id=30,
        timestamp=0.0,
        opencv=[cv_det],
        yolo=[yolo_det],
        yolo_ran=True,
        yolo_frame_id=1,
        yolo_age_frames=config.MAX_YOLO_RESULT_AGE_FRAMES + 1,
    )

    selected = adapter.fuse_packet(packet).selected

    assert selected["yolo_verified"] is False
    assert selected["yolo_fresh"] is False


def test_release_gate_does_not_advance_state_twice_in_one_packet():
    adapter = FusionAdapter("red_square", [])
    state_machine = adapter.state_machine_by_target["red_square"]
    state_machine.cv_seen_counter = config.CANDIDATE_MIN_FRAMES
    state_machine.lock_counter = config.LOCK_MIN_FRAMES - 1
    cv_det = {
        "source": "opencv",
        "target_type": "red_square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.95,
        "state": "DETECTED",
        "frame_id": 50,
        "error": (0.0, 0.0),
    }
    yolo_det = {
        "source": "yolo",
        "target_type": "square",
        "bbox": (10, 10, 20, 20),
        "center": (20, 20),
        "confidence": 0.95,
        "state": "DETECTED",
        "frame_id": 50,
    }
    packet = RawDetectionPacket(
        frame_id=50,
        timestamp=0.0,
        opencv=[cv_det],
        yolo=[yolo_det],
        yolo_ran=True,
        yolo_frame_id=50,
        yolo_age_frames=0,
    )

    selected = adapter.fuse_packet(packet).selected

    assert selected["target_state"] == LOCKED
    assert selected["release_gate"] is True
    assert state_machine.lock_counter == config.LOCK_MIN_FRAMES
