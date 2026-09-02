from teknofest_iha.core.coordinate_frame import CoordinateFrameMapper


def test_identity_frame_keeps_local_xy():
    mapper = CoordinateFrameMapper("identity")
    assert mapper.nav_xy_from_local(1.0, 2.0) == (1.0, 2.0)
    assert mapper.local_velocity_from_nav(0.3, -0.4) == (0.3, -0.4)


def test_gazebo_swapped_frame_matches_measured_sitl_mapping():
    mapper = CoordinateFrameMapper("gazebo_xy_swapped")
    assert mapper.nav_xy_from_local(52.1, 3.7) == (3.7, 52.1)
    assert mapper.local_velocity_from_nav(0.0, 0.7) == (0.7, 0.0)
    assert mapper.nav_velocity_from_local(0.7, 0.0) == (0.0, 0.7)
