from __future__ import annotations

"""Thin MAVLink adapter used by the ROS bridge node.

The adapter hides pymavlink connection handling and ArduPilot command details.
It intentionally exposes simple methods such as `arm`, `takeoff`,
`send_velocity`, and `drop_payload` so mission logic remains protocol-agnostic.
"""

import math
import time
from dataclasses import dataclass

from teknofest_iha.interfaces.drone_models import Altitude, DroneState, LocalPosition

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - handled at runtime on ROS machine
    mavutil = None


@dataclass
class MavlinkStatus:
    state: DroneState
    local_position: LocalPosition
    altitude: Altitude


class MavlinkAdapter:
    """ArduPilot MAVLink transport adapter. It contains no mission/perception logic."""

    def __init__(self, connection: str, source_system: int = 255, heartbeat_timeout_s: float = 30.0) -> None:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed. Install it in the ROS environment.")
        self.connection = connection
        self.source_system = int(source_system)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.master = None
        self.state = DroneState()
        self.local_position = LocalPosition()
        self.altitude = Altitude()
        self.last_gcs_heartbeat_s = 0.0

    def connect(self) -> DroneState:
        self.master = mavutil.mavlink_connection(
            self.connection,
            source_system=self.source_system,
            autoreconnect=True,
        )
        heartbeat = self.master.wait_heartbeat(timeout=self.heartbeat_timeout_s)
        if heartbeat is None:
            raise TimeoutError(f"MAVLink heartbeat timeout after {self.heartbeat_timeout_s:.1f}s")
        self._update_heartbeat(heartbeat)
        self._send_gcs_heartbeat(force=True)
        self._request_data_streams()
        return self.state

    def set_mode(self, mode: str, timeout_s: float = 5.0) -> bool:
        self._require_master()
        self._send_gcs_heartbeat()
        mode_mapping = self.master.mode_mapping()
        mode_id = mode_mapping.get(mode)
        if mode_id is None:
            raise RuntimeError(f"Mode {mode!r} is not available. Modes: {sorted(mode_mapping)}")
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self.read_messages(timeout_s=0.2)
            if self.state.mode == mode:
                return True
        return False

    def arm(self, timeout_s: float = 10.0) -> bool:
        self._require_master()
        self._send_gcs_heartbeat()
        self.master.arducopter_arm()
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self.read_messages(timeout_s=0.2)
            if self.state.armed:
                return True
        return False

    def disarm(self) -> None:
        self._require_master()
        self.master.arducopter_disarm()

    def takeoff(self, altitude_m: float) -> None:
        self.command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, altitude_m)

    def land(self) -> None:
        try:
            self.set_mode("LAND", timeout_s=2.0)
        except Exception:
            self.command_long(mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0)

    def rtl(self) -> None:
        self.set_mode("RTL", timeout_s=2.0)

    def send_velocity_ned(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        self._require_master()
        self._send_gcs_heartbeat()
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )
        if abs(yaw_rate) <= 1e-6:
            type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0,
            0,
            0,
            float(vx),
            float(vy),
            float(vz),
            0,
            0,
            0,
            0,
            float(yaw_rate),
        )

    def send_position_ned(self, x: float, y: float, z: float, yaw: float | None = None) -> None:
        self._require_master()
        self._send_gcs_heartbeat()
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        if yaw is None:
            type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            yaw = 0.0
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            float(x),
            float(y),
            float(z),
            0,
            0,
            0,
            0,
            0,
            0,
            float(yaw),
            0,
        )

    def drop_payload(self, servo: int, pwm: int, hold_seconds: float = 0.8, reset_pwm: int | None = None) -> None:
        self.command_long(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, servo, pwm, 0, 0, 0, 0, 0)
        time.sleep(max(0.0, hold_seconds))
        if reset_pwm is not None:
            self.command_long(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, servo, reset_pwm, 0, 0, 0, 0, 0)

    def command_long(self, command: int, p1: float, p2: float, p3: float, p4: float, p5: float, p6: float, p7: float) -> None:
        self._require_master()
        self._send_gcs_heartbeat()
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command,
            0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
        )

    def read_messages(self, timeout_s: float = 0.0) -> MavlinkStatus:
        self._require_master()
        self._send_gcs_heartbeat()
        end = time.monotonic() + max(0.0, timeout_s)
        while True:
            blocking = timeout_s > 0.0 and time.monotonic() < end
            msg = self.master.recv_match(blocking=blocking, timeout=max(0.0, end - time.monotonic()) if blocking else 0)
            if msg is None:
                break
            self._handle_message(msg)
            if not blocking:
                continue
            if time.monotonic() >= end:
                break
        return MavlinkStatus(self.state, self.local_position, self.altitude)

    def _handle_message(self, msg) -> None:
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            self._update_heartbeat(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            self.local_position = LocalPosition(
                x=float(msg.x),
                y=float(msg.y),
                z=float(msg.z),
                vx=float(msg.vx),
                vy=float(msg.vy),
                vz=float(msg.vz),
                frame="NED",
            )
        elif msg_type == "GLOBAL_POSITION_INT":
            self.altitude = Altitude(
                relative_m=float(msg.relative_alt) / 1000.0,
                amsl_m=float(msg.alt) / 1000.0,
            )

    def _update_heartbeat(self, msg) -> None:
        mode = mavutil.mode_string_v10(msg)
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self.state = DroneState(
            connected=True,
            armed=armed,
            mode=mode,
            system_id=getattr(self.master, "target_system", None),
            component_id=getattr(self.master, "target_component", None),
            last_heartbeat_s=time.time(),
        )

    def _require_master(self) -> None:
        if self.master is None:
            raise RuntimeError("MAVLink connection is not open.")

    def _send_gcs_heartbeat(self, force: bool = False) -> None:
        self._require_master()
        now = time.monotonic()
        if not force and now - self.last_gcs_heartbeat_s < 1.0:
            return
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )
        self.last_gcs_heartbeat_s = now

    def _request_data_streams(self) -> None:
        self._require_master()
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,
            1,
        )


def yaw_deg_to_rad(yaw_deg: float) -> float:
    return math.radians(yaw_deg)
