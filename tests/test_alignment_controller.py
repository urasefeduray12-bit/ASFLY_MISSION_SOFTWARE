from teknofest_iha.core.alignment_controller import AlignmentController


def test_alignment_velocity_uses_navigation_frame_axes():
    controller = AlignmentController(640, 480, 40, 0.2)

    vx, vy = controller.velocity_from_center((320, 80))
    assert vx > 0.0
    assert abs(vy) < 1e-9

    vx, vy = controller.velocity_from_center((480, 240))
    assert abs(vx) < 1e-9
    assert vy < 0.0


def test_alignment_velocity_flips_with_westbound_lane():
    controller = AlignmentController(640, 480, 40, 0.2)

    vx, vy = controller.velocity_from_center((320, 80), forward_sign=-1.0)
    assert vx < 0.0
    assert abs(vy) < 1e-9

    vx, vy = controller.velocity_from_center((480, 240), forward_sign=-1.0)
    assert abs(vx) < 1e-9
    assert vy > 0.0
