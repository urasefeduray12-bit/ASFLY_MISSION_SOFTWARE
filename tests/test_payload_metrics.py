from teknofest_iha.core.payload_metrics import TargetSpec, estimate_payload_drop


def test_payload_drop_estimate_uses_release_velocity_and_altitude():
    target = TargetSpec(center_x=10.0, center_y=0.0, size_x=2.0, size_y=2.0)

    estimate = estimate_payload_drop(
        release_x=9.0,
        release_y=0.0,
        release_altitude_m=4.903325,
        release_vx=1.0,
        release_vy=0.0,
        target=target,
    )

    assert abs(estimate.fall_time_s - 1.0) < 1e-4
    assert abs(estimate.estimated_impact_x - 10.0) < 1e-4
    assert estimate.distance_to_center_m < 1e-4
    assert estimate.inside_target_footprint is True


def test_payload_drop_estimate_reports_miss_distance():
    target = TargetSpec(center_x=45.0, center_y=4.0, size_x=1.0, size_y=1.0)

    estimate = estimate_payload_drop(
        release_x=46.0,
        release_y=4.0,
        release_altitude_m=0.0,
        release_vx=0.0,
        release_vy=0.0,
        target=target,
    )

    assert estimate.distance_to_center_m == 1.0
    assert estimate.inside_target_footprint is False
