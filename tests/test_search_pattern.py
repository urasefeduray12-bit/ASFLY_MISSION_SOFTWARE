from teknofest_iha.core.search_pattern import LawnmowerSearchPattern


def test_lawnmower_waypoints_alternate_direction():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)
    assert pattern.waypoints() == [
        (0.0, -5.0),
        (10.0, -5.0),
        (10.0, 0.0),
        (0.0, 0.0),
        (0.0, 5.0),
        (10.0, 5.0),
    ]


def test_lawnmower_advances_after_passing_lane_endpoint():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    index, vx, vy = pattern.next_velocity(11.0, -5.0, 1, 2.0)

    assert index == 2
    assert vx == 0.0
    assert vy > 0.0


def test_lawnmower_advances_at_lane_endpoint_with_cross_track_error():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    index, vx, vy = pattern.next_velocity(9.2, -3.0, 1, 2.0)

    assert index == 2
    assert vx == 0.0
    assert vy > 0.0


def test_lawnmower_does_not_return_along_completed_lane():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    index, vx, vy = pattern.next_velocity(10.5, -0.1, 2, 2.0)

    assert index == 3
    assert vx < 0.0
    assert vy == 0.0


def test_lawnmower_horizontal_lane_has_no_lateral_velocity():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    index, vx, vy = pattern.next_velocity(4.0, -4.2, 1, 2.0)

    assert index == 1
    assert vx > 0.0
    assert vy == 0.0


def test_lawnmower_can_start_from_nearest_x_max_side():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    assert pattern.waypoints_from_nearest_start(9.0, -8.0) == [
        (10.0, -5.0),
        (0.0, -5.0),
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (0.0, 5.0),
    ]


def test_nearest_start_avoids_crossing_to_far_lane_start():
    pattern = LawnmowerSearchPattern(0.0, 10.0, -5.0, 5.0, 5.0)

    index, vx, vy = pattern.next_velocity_from_start(10.0, -5.0, 0, 2.0, 0.5, start_from_x_max=True)

    assert index == 1
    assert vx < 0.0
    assert vy == 0.0
