from __future__ import annotations

"""ROS 2 mission orchestration process.

This is the only node that turns perception/fusion state into vehicle intent.
It reads target confidence, vehicle telemetry, safety status, and mission start
commands, then publishes high-level drone command JSON messages.

The module deliberately keeps MAVLink details out of the mission logic; those
details live in `mavlink_bridge_node` and `MavlinkAdapter`.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_iha.core.alignment_controller import AlignmentController
from teknofest_iha.core.coordinate_frame import CoordinateFrameMapper
from teknofest_iha.core.geofence import Geofence
from teknofest_iha.core.mission_states import MissionState
from teknofest_iha.core.payload_controller import PayloadController
from teknofest_iha.core.payload_metrics import TargetSpec, estimate_payload_drop
from teknofest_iha.core.search_pattern import LawnmowerSearchPattern
from teknofest_iha.core.state_machine import MissionInputs, MissionStateMachine
from teknofest_iha.core.target_selection import choose_visible_unreleased_target
from teknofest_iha.interfaces.detection_models import FusedTargetPacket
from teknofest_iha.interfaces.drone_models import Altitude, DroneState, LocalPosition, command_json


class MissionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_manager_node")
        self.declare_parameter("primary_target", "blue_square")
        self.declare_parameter("secondary_targets", ["red_square"])
        self.declare_parameter("takeoff_altitude_m", 10.0)
        self.declare_parameter("altitude_tolerance_m", 0.5)
        self.declare_parameter("restore_altitude_tolerance_m", 0.2)
        self.declare_parameter("end_action", "land")
        self.declare_parameter("post_drop_hover_s", 3.0)
        self.declare_parameter("lock_seconds", 1.5)
        self.declare_parameter("lock_min_confidence", 0.60)
        self.declare_parameter("approach_enabled", True)
        self.declare_parameter("drop_altitude_m", 2.0)
        self.declare_parameter("descend_speed_mps", 0.20)
        self.declare_parameter("climb_speed_mps", 0.35)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("search_speed_mps", 2.0)
        self.declare_parameter("search_acceptance_radius_m", 0.5)
        self.declare_parameter("lane_spacing_m", 5.0)
        self.declare_parameter("align_max_speed_mps", 0.6)
        self.declare_parameter("center_tolerance_px", 40)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("search_x_min", 0.0)
        self.declare_parameter("search_x_max", 100.0)
        self.declare_parameter("search_y_min", -15.0)
        self.declare_parameter("search_y_max", 15.0)
        self.declare_parameter("payload_dry_run", True)
        self.declare_parameter("payload_servo", 9)
        self.declare_parameter("payload_servo_map_json", '{"blue_square":9,"red_square":10}')
        self.declare_parameter("payload_pwm", 1900)
        self.declare_parameter("payload_reset_pwm", 1100)
        self.declare_parameter("payload_hold_seconds", 0.8)
        self.declare_parameter("coordinate_frame", "identity")
        self.declare_parameter("autostart", True)
        self.declare_parameter(
            "target_specs_json",
            '{"blue_square":{"center":[62.0,-5.0],"size":[2.0,2.0]},"red_square":{"center":[45.0,4.0],"size":[1.0,1.0]}}',
        )

        self.primary_target = str(self.get_parameter("primary_target").value)
        secondary_targets = [str(target) for target in list(self.get_parameter("secondary_targets").value)]
        self.target_sequence = tuple(dict.fromkeys([self.primary_target, *secondary_targets]))
        self.takeoff_altitude_m = float(self.get_parameter("takeoff_altitude_m").value)
        self.lock_seconds = float(self.get_parameter("lock_seconds").value)
        self.lock_min_confidence = float(self.get_parameter("lock_min_confidence").value)
        self.approach_enabled = bool(self.get_parameter("approach_enabled").value)
        self.drop_altitude_m = float(self.get_parameter("drop_altitude_m").value)
        self.descend_speed_mps = float(self.get_parameter("descend_speed_mps").value)
        self.climb_speed_mps = float(self.get_parameter("climb_speed_mps").value)
        self.restore_altitude_tolerance_m = float(self.get_parameter("restore_altitude_tolerance_m").value)
        self.search_acceptance_radius_m = float(self.get_parameter("search_acceptance_radius_m").value)
        self.payload_dry_run = bool(self.get_parameter("payload_dry_run").value)
        self.payload_servo_by_target = self._load_payload_servo_map()
        self.state_machine = MissionStateMachine(
            takeoff_altitude_m=self.takeoff_altitude_m,
            altitude_tolerance_m=float(self.get_parameter("altitude_tolerance_m").value),
            post_drop_hover_s=float(self.get_parameter("post_drop_hover_s").value),
            target_sequence=self.target_sequence,
            restore_altitude_tolerance_m=self.restore_altitude_tolerance_m,
        )
        self.payload = PayloadController()
        self.target_specs = self._load_target_specs()
        self.alignment = AlignmentController(
            int(self.get_parameter("image_width").value),
            int(self.get_parameter("image_height").value),
            float(self.get_parameter("center_tolerance_px").value),
            float(self.get_parameter("align_max_speed_mps").value),
        )
        self.search = LawnmowerSearchPattern(
            float(self.get_parameter("search_x_min").value),
            float(self.get_parameter("search_x_max").value),
            float(self.get_parameter("search_y_min").value),
            float(self.get_parameter("search_y_max").value),
            float(self.get_parameter("lane_spacing_m").value),
        )
        self.geofence = Geofence(
            float(self.get_parameter("search_x_min").value),
            float(self.get_parameter("search_x_max").value),
            float(self.get_parameter("search_y_min").value),
            float(self.get_parameter("search_y_max").value),
        )
        self.frame_mapper = CoordinateFrameMapper(str(self.get_parameter("coordinate_frame").value))

        self.drone_state = DroneState()
        self.local_position = LocalPosition()
        self.altitude = Altitude()
        self.safety_status = "OK"
        self.last_fusion: FusedTargetPacket | None = None
        self.lock_started_at: float | None = None
        self.lock_target: str | None = None
        self.search_index = 0
        self.search_start_from_x_max: bool | None = None
        self.commanded_states: set[str] = set()
        self.command_publish_times: dict[str, float] = {}
        self.alignment_forward_sign = 1.0
        self.mission_started = bool(self.get_parameter("autostart").value)

        self.mode_pub = self.create_publisher(String, "/drone/cmd_mode", 10)
        self.arm_pub = self.create_publisher(String, "/drone/cmd_arm", 10)
        self.takeoff_pub = self.create_publisher(String, "/drone/cmd_takeoff", 10)
        self.velocity_pub = self.create_publisher(String, "/drone/cmd_velocity", 10)
        self.land_pub = self.create_publisher(String, "/drone/cmd_land", 10)
        self.drop_pub = self.create_publisher(String, "/drone/cmd_drop", 10)
        self.state_pub = self.create_publisher(String, "/mission/state", 10)
        self.event_pub = self.create_publisher(String, "/mission/event", 10)

        self.create_subscription(String, "/fusion/target", self.on_fusion, 10)
        self.create_subscription(String, "/drone/state", self.on_drone_state, 10)
        self.create_subscription(String, "/drone/local_position", self.on_local_position, 10)
        self.create_subscription(String, "/drone/altitude", self.on_altitude, 10)
        self.create_subscription(String, "/safety/status", self.on_safety, 10)
        self.create_subscription(String, "/mission/cmd_start", self.on_start_command, 10)
        rate = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / max(1.0, rate), self.on_timer)

    def on_fusion(self, msg: String) -> None:
        self.last_fusion = FusedTargetPacket.from_json(msg.data)

    def on_drone_state(self, msg: String) -> None:
        self.drone_state = DroneState.from_json(msg.data)

    def on_local_position(self, msg: String) -> None:
        self.local_position = LocalPosition.from_json(msg.data)

    def on_altitude(self, msg: String) -> None:
        self.altitude = Altitude.from_json(msg.data)

    def on_safety(self, msg: String) -> None:
        self.safety_status = str(json.loads(msg.data).get("status", "OK"))

    def on_start_command(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command in {"start", "true", "1", "go"}:
            self.mission_started = True
            self.get_logger().info("Mission start command accepted")

    def _load_target_specs(self) -> dict[str, TargetSpec]:
        raw = str(self.get_parameter("target_specs_json").value)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"target_specs_json must be valid JSON: {exc}") from exc
        specs: dict[str, TargetSpec] = {}
        for target_type, spec in data.items():
            center = spec["center"]
            size = spec["size"]
            specs[str(target_type)] = TargetSpec(
                center_x=float(center[0]),
                center_y=float(center[1]),
                size_x=float(size[0]),
                size_y=float(size[1]),
            )
        return specs

    def _load_payload_servo_map(self) -> dict[str, int]:
        raw = str(self.get_parameter("payload_servo_map_json").value)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"payload_servo_map_json must be valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("payload_servo_map_json must be a JSON object")
        return {str(target_type): int(servo) for target_type, servo in data.items()}

    def _payload_servo_for_target(self, target_type: str) -> int:
        fallback_servo = int(self.get_parameter("payload_servo").value)
        return self.payload_servo_by_target.get(target_type, fallback_servo)

    def on_timer(self) -> None:
        if not self.mission_started:
            self._publish_waiting_for_start()
            return
        self._promote_visible_target()
        active_target = self.state_machine.active_target or self.primary_target
        selected = self._selected_target(active_target)
        centered = self._target_centered(selected)
        lock_stable = self._target_locked(selected, centered)
        drop_done = not self.payload.can_release(active_target)
        relative_altitude_m = self._relative_altitude_m()
        drop_altitude_reached = relative_altitude_m <= self.drop_altitude_m
        drop_ready = lock_stable and (not self.approach_enabled or drop_altitude_reached)
        inputs = MissionInputs(
            camera_ready=self.last_fusion is not None,
            mavlink_connected=self.drone_state.connected,
            guided=self.drone_state.mode == "GUIDED",
            armed=self.drone_state.armed,
            altitude_m=relative_altitude_m,
            target=selected,
            target_centered=centered,
            target_locked=drop_ready,
            safety_level=self.safety_status,
            drop_done=drop_done,
            return_confirmed=self.drone_state.mode == "RTL",
        )
        state = self.state_machine.update(inputs)
        active_target = self.state_machine.active_target or active_target
        self._publish_state(state, active_target, relative_altitude_m)
        self._act_for_state(state, active_target, selected, centered, drop_ready, lock_stable)

    def _publish_waiting_for_start(self) -> None:
        self.state_pub.publish(
            String(
                data=json.dumps(
                    {
                        "state": "WAIT_START",
                        "active_target": self.state_machine.active_target or self.primary_target,
                        "altitude_m": self._relative_altitude_m(),
                    }
                )
            )
        )

    def _promote_visible_target(self) -> None:
        visible_target = self._visible_unreleased_target()
        if visible_target is None:
            return
        active_target = self.state_machine.active_target
        if visible_target == active_target:
            return
        if self.state_machine.state in {MissionState.SEARCH_TARGET, MissionState.TARGET_CANDIDATE}:
            self._move_target_to_active_slot(visible_target)
            return
        if self.state_machine.state in {MissionState.TARGET_ALIGN, MissionState.TARGET_VERIFY}:
            if active_target is not None and self._selected_target(active_target) is not None:
                return
            self._move_target_to_active_slot(visible_target)

    def _visible_unreleased_target(self) -> str | None:
        if self.last_fusion is None:
            return None
        return choose_visible_unreleased_target(
            self.last_fusion.selected,
            self.last_fusion.targets,
            self.target_sequence,
            self.payload.can_release,
        )

    def _move_target_to_active_slot(self, target_type: str) -> None:
        if target_type not in self.target_sequence:
            return
        active_index = self.state_machine.active_target_index
        remaining = list(self.state_machine.target_sequence[active_index:])
        if target_type not in remaining:
            return
        remaining.remove(target_type)
        prefix = list(self.state_machine.target_sequence[:active_index])
        self.state_machine.target_sequence = tuple(prefix + [target_type] + remaining)

    def _selected_target(self, target_type: str) -> dict | None:
        if self.last_fusion is None:
            return None
        selected = self.last_fusion.selected
        if selected and selected.get("target_type") == target_type:
            return selected
        for target in self.last_fusion.targets:
            if target.get("target_type") == target_type:
                return target
        return None

    def _relative_altitude_m(self) -> float:
        if self.local_position.z < -0.05:
            return max(0.0, -self.local_position.z)
        return self.altitude.relative_m

    def _target_centered(self, selected: dict | None) -> bool:
        if selected is None or selected.get("center") is None:
            return False
        return self.alignment.is_centered(tuple(selected["center"]))

    def _target_locked(self, selected: dict | None, centered: bool) -> bool:
        if selected is None:
            self.lock_started_at = None
            self.lock_target = None
            return False
        target_state = str(selected.get("target_state", selected.get("state", "")))
        target_type = str(selected.get("target_type", ""))
        confidence = float(selected.get("fusion_confidence", selected.get("confidence", 0.0)))
        if target_state in ("TRACKING", "LOCKED", "DROP_READY") and centered and confidence >= self.lock_min_confidence:
            if self.lock_target != target_type:
                self.lock_started_at = None
                self.lock_target = target_type
            self.lock_started_at = self.lock_started_at or time.time()
            return time.time() - self.lock_started_at >= self.lock_seconds
        self.lock_started_at = None
        self.lock_target = None
        return False

    def _act_for_state(
        self,
        state: MissionState,
        active_target: str,
        selected: dict | None,
        centered: bool,
        locked: bool,
        lock_stable: bool,
    ) -> None:
        name = state.value
        if state == MissionState.SET_GUIDED:
            self._publish_periodic(name, self.mode_pub, command_json("set_mode", mode="GUIDED"))
        elif state == MissionState.ARM:
            if self.drone_state.mode != "GUIDED":
                self._publish_periodic("ARM_SET_GUIDED", self.mode_pub, command_json("set_mode", mode="GUIDED"))
            self._publish_periodic(name, self.arm_pub, command_json("arm", arm=True))
        elif state == MissionState.TAKEOFF:
            if self.drone_state.mode != "GUIDED":
                self._publish_periodic("TAKEOFF_SET_GUIDED", self.mode_pub, command_json("set_mode", mode="GUIDED"))
            if not self.drone_state.armed:
                self._publish_periodic("TAKEOFF_ARM", self.arm_pub, command_json("arm", arm=True))
            self._publish_periodic(name, self.takeoff_pub, command_json("takeoff", altitude_m=self.takeoff_altitude_m))
        elif state == MissionState.SEARCH_TARGET:
            nav_x, nav_y = self.frame_mapper.nav_xy_from_local(self.local_position.x, self.local_position.y)
            if self.search_start_from_x_max is None:
                self.search_start_from_x_max = self.search.start_from_x_max_is_nearest(nav_x, nav_y)
            self.search_index, vx, vy = self.search.next_velocity_from_start(
                nav_x,
                nav_y,
                self.search_index,
                float(self.get_parameter("search_speed_mps").value),
                self.search_acceptance_radius_m,
                self.search_start_from_x_max,
            )
            if abs(vx) >= abs(vy) and abs(vx) > 0.05:
                self.alignment_forward_sign = 1.0 if vx >= 0.0 else -1.0
            vx, vy = self.geofence.clamp_velocity(nav_x, nav_y, vx, vy)
            local_vx, local_vy = self.frame_mapper.local_velocity_from_nav(vx, vy)
            self.velocity_pub.publish(String(data=command_json("velocity", vx=local_vx, vy=local_vy, vz=self._restore_search_altitude_vz())))
        elif state in (MissionState.TARGET_CANDIDATE, MissionState.TARGET_ALIGN, MissionState.TARGET_VERIFY):
            if selected is not None:
                vx, vy = self.alignment.velocity_from_center(tuple(selected["center"]), self.alignment_forward_sign)
                local_vx, local_vy = self.frame_mapper.local_velocity_from_nav(vx, vy)
                vz = self._target_approach_vz(state, centered, lock_stable)
                self.velocity_pub.publish(String(data=command_json("velocity", vx=local_vx, vy=local_vy, vz=vz)))
        elif state == MissionState.DROP_TARGET and selected is not None and centered and locked:
            if self.payload.can_release(active_target):
                self.payload.mark_released(active_target)
                event = command_json(
                    "drop_payload",
                    target_type=active_target,
                    dry_run=self.payload_dry_run,
                    servo=self._payload_servo_for_target(active_target),
                    pwm=int(self.get_parameter("payload_pwm").value),
                    reset_pwm=int(self.get_parameter("payload_reset_pwm").value),
                    hold_seconds=float(self.get_parameter("payload_hold_seconds").value),
                )
                estimate = self._drop_estimate(active_target)
                if estimate is not None:
                    event_data = json.loads(event)
                    event_data["drop_estimate"] = estimate.as_dict()
                    event = json.dumps(event_data, separators=(",", ":"))
                self.event_pub.publish(String(data=event))
                self.drop_pub.publish(String(data=event))
        elif state == MissionState.POST_DROP_HOVER:
            self.velocity_pub.publish(
                String(data=command_json("velocity", vx=0.0, vy=0.0, vz=self._restore_search_altitude_vz()))
            )
        elif state == MissionState.RETURN_HOME:
            self._publish_periodic(name, self.mode_pub, command_json("set_mode", mode="RTL"))
        elif state == MissionState.LAND:
            self._publish_periodic(name, self.land_pub, command_json("land"))
        elif state == MissionState.FAILSAFE:
            self.velocity_pub.publish(String(data=command_json("velocity", vx=0.0, vy=0.0, vz=0.0)))
            self._publish_periodic(name, self.land_pub, command_json("failsafe_land"))

    def _target_approach_vz(self, state: MissionState, centered: bool, lock_stable: bool) -> float:
        if not self.approach_enabled:
            return 0.0
        if state != MissionState.TARGET_VERIFY:
            return 0.0
        if not centered or not lock_stable:
            return 0.0
        if self._relative_altitude_m() <= self.drop_altitude_m:
            return 0.0
        return max(0.0, self.descend_speed_mps)

    def _restore_search_altitude_vz(self) -> float:
        if self._relative_altitude_m() >= self.takeoff_altitude_m - self.restore_altitude_tolerance_m:
            return 0.0
        return -max(0.0, self.climb_speed_mps)

    def _publish_once(self, key: str, publisher, payload: str) -> None:
        if key in self.commanded_states:
            return
        self.commanded_states.add(key)
        publisher.publish(String(data=payload))

    def _publish_periodic(self, key: str, publisher, payload: str, period_s: float = 1.0) -> None:
        now = time.monotonic()
        last = self.command_publish_times.get(key, 0.0)
        if now - last < period_s:
            return
        self.command_publish_times[key] = now
        publisher.publish(String(data=payload))

    def _publish_state(self, state: MissionState, active_target: str, altitude_m: float) -> None:
        nav_x, nav_y = self.frame_mapper.nav_xy_from_local(self.local_position.x, self.local_position.y)
        search_status = self._search_status()
        payload = {
            "state": state.value,
            "active_target": active_target,
            "target_sequence": list(self.state_machine.target_sequence),
            "released_targets": sorted(self.payload.released_targets),
            "altitude_m": altitude_m,
            "local_x": self.local_position.x,
            "local_y": self.local_position.y,
            "nav_x": nav_x,
            "nav_y": nav_y,
            "coordinate_frame": self.frame_mapper.mode,
            "alignment_forward_sign": self.alignment_forward_sign,
            "timestamp": time.time(),
            **search_status,
        }
        self.state_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _search_status(self) -> dict:
        start_from_x_max = bool(self.search_start_from_x_max) if self.search_start_from_x_max is not None else False
        points = self.search.waypoints_from_start(start_from_x_max)
        if not points:
            return {"search_index": self.search_index, "search_axis": "-", "search_target": None}
        index = min(max(self.search_index, 0), len(points) - 1)
        tx, ty = points[index]
        axis = "initial"
        if index > 0:
            px, py = points[index - 1]
            axis = "x" if abs(tx - px) >= abs(ty - py) else "y"
        return {
            "search_index": index,
            "search_axis": axis,
            "search_target": [tx, ty],
            "search_start": "x_max" if start_from_x_max else "x_min",
        }

    def _drop_estimate(self, target_type: str):
        target = self.target_specs.get(target_type)
        if target is None:
            return None
        nav_x, nav_y = self.frame_mapper.nav_xy_from_local(self.local_position.x, self.local_position.y)
        nav_vx, nav_vy = self.frame_mapper.nav_velocity_from_local(self.local_position.vx, self.local_position.vy)
        return estimate_payload_drop(
            release_x=nav_x,
            release_y=nav_y,
            release_altitude_m=self._relative_altitude_m(),
            release_vx=nav_vx,
            release_vy=nav_vy,
            target=target,
        )


def main() -> None:
    rclpy.init()
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
