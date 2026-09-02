from teknofest_iha.core.mission_states import MissionState
from teknofest_iha.core.state_machine import MissionInputs, MissionStateMachine


def test_initial_progression_to_connect_mavlink():
    machine = MissionStateMachine(10.0, 0.5, 1.0)
    assert machine.update(MissionInputs()) == MissionState.WAIT_FOR_CAMERA
    assert machine.update(MissionInputs(camera_ready=True)) == MissionState.CONNECT_MAVLINK


def test_two_target_payload_sequence_returns_home():
    machine = MissionStateMachine(10.0, 0.5, 1.0, ("blue_square", "red_square"))

    assert machine.update(MissionInputs()) == MissionState.WAIT_FOR_CAMERA
    assert machine.update(MissionInputs(camera_ready=True)) == MissionState.CONNECT_MAVLINK
    assert machine.update(MissionInputs(camera_ready=True, mavlink_connected=True)) == MissionState.SET_GUIDED
    assert machine.update(MissionInputs(camera_ready=True, mavlink_connected=True, guided=True)) == MissionState.ARM
    assert machine.update(MissionInputs(camera_ready=True, mavlink_connected=True, guided=True, armed=True)) == MissionState.TAKEOFF
    assert machine.update(MissionInputs(camera_ready=True, mavlink_connected=True, guided=True, armed=True, altitude_m=10.0)) == MissionState.SEARCH_TARGET
    assert machine.active_target == "blue_square"

    locked_blue = MissionInputs(target={"target_type": "blue_square"}, target_centered=True, target_locked=True)
    assert machine.update(locked_blue) == MissionState.TARGET_CANDIDATE
    assert machine.update(locked_blue) == MissionState.TARGET_ALIGN
    assert machine.update(locked_blue) == MissionState.TARGET_VERIFY
    assert machine.update(locked_blue) == MissionState.DROP_TARGET
    assert machine.update(MissionInputs(drop_done=True)) == MissionState.POST_DROP_HOVER
    machine.entered_at -= 2.0
    assert machine.update(MissionInputs(altitude_m=2.0)) == MissionState.POST_DROP_HOVER
    assert machine.update(MissionInputs(altitude_m=10.0)) == MissionState.SEARCH_TARGET
    assert machine.active_target == "red_square"

    locked_red = MissionInputs(target={"target_type": "red_square"}, target_centered=True, target_locked=True)
    assert machine.update(locked_red) == MissionState.TARGET_CANDIDATE
    assert machine.update(locked_red) == MissionState.TARGET_ALIGN
    assert machine.update(locked_red) == MissionState.TARGET_VERIFY
    assert machine.update(locked_red) == MissionState.DROP_TARGET
    assert machine.update(MissionInputs(drop_done=True)) == MissionState.POST_DROP_HOVER
    machine.entered_at -= 2.0
    assert machine.update(MissionInputs(altitude_m=2.0)) == MissionState.POST_DROP_HOVER
    assert machine.update(MissionInputs(altitude_m=10.0)) == MissionState.RETURN_HOME
    assert machine.update(MissionInputs(return_confirmed=True)) == MissionState.MISSION_COMPLETE


def test_candidate_enters_alignment_before_target_is_locked():
    machine = MissionStateMachine(10.0, 0.5, 1.0, ("blue_square",))
    machine.state = MissionState.SEARCH_TARGET

    visible_but_not_locked = MissionInputs(target={"target_type": "blue_square"})

    assert machine.update(visible_but_not_locked) == MissionState.TARGET_CANDIDATE
    assert machine.update(visible_but_not_locked) == MissionState.TARGET_ALIGN


def test_post_drop_requires_restore_altitude_tolerance():
    machine = MissionStateMachine(
        10.0,
        0.5,
        1.0,
        ("blue_square", "red_square"),
        restore_altitude_tolerance_m=0.2,
    )
    machine.state = MissionState.POST_DROP_HOVER
    machine.entered_at -= 2.0

    assert machine.update(MissionInputs(altitude_m=9.6)) == MissionState.POST_DROP_HOVER
    assert machine.update(MissionInputs(altitude_m=9.85)) == MissionState.SEARCH_TARGET
