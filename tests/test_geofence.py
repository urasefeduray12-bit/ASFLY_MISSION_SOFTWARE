from teknofest_iha.core.geofence import Geofence, GeofenceLevel


def test_geofence_levels():
    geofence = Geofence(0.0, 100.0, -15.0, 15.0, warning_margin_m=2.0)
    assert geofence.check(50.0, 0.0) == GeofenceLevel.OK
    assert geofence.check(1.0, 0.0) == GeofenceLevel.WARNING
    assert geofence.check(101.0, 0.0) == GeofenceLevel.VIOLATION


def test_velocity_clamp_blocks_outward_motion():
    geofence = Geofence(0.0, 100.0, -15.0, 15.0, hard_margin_m=0.5)
    assert geofence.clamp_velocity(0.2, 0.0, -1.0, 0.0) == (0.0, 0.0)
    assert geofence.clamp_velocity(50.0, 0.0, -1.0, 0.5) == (-1.0, 0.5)
