from __future__ import annotations

import json
import time
from collections import Counter

import config
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_iha.interfaces.drone_models import DroneState, LocalPosition


class MissionConsoleNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_console_node")
        self.declare_parameter("print_rate_hz", 5.0)

        self.raw_packet: dict = {}
        self.fusion_packet: dict = {}
        self.mission_state: dict = {}
        self.last_event: dict = {}
        self.safety_status = "OK"
        self.drone_state = DroneState()
        self.local_position = LocalPosition()
        self.last_printed_event = ""

        self.create_subscription(String, "/perception/raw_detections", self.on_raw, 10)
        self.create_subscription(String, "/fusion/target", self.on_fusion, 10)
        self.create_subscription(String, "/mission/state", self.on_mission_state, 10)
        self.create_subscription(String, "/mission/event", self.on_mission_event, 10)
        self.create_subscription(String, "/drone/state", self.on_drone_state, 10)
        self.create_subscription(String, "/drone/local_position", self.on_local_position, 10)
        self.create_subscription(String, "/safety/status", self.on_safety, 10)

        rate = max(0.2, float(self.get_parameter("print_rate_hz").value))
        self.create_timer(1.0 / rate, self.on_timer)

    def on_raw(self, msg: String) -> None:
        self.raw_packet = self._json_or_empty(msg.data)

    def on_fusion(self, msg: String) -> None:
        self.fusion_packet = self._json_or_empty(msg.data)

    def on_mission_state(self, msg: String) -> None:
        self.mission_state = self._json_or_empty(msg.data)

    def on_mission_event(self, msg: String) -> None:
        self.last_event = self._json_or_empty(msg.data)

    def on_drone_state(self, msg: String) -> None:
        self.drone_state = DroneState.from_json(msg.data)

    def on_local_position(self, msg: String) -> None:
        self.local_position = LocalPosition.from_json(msg.data)

    def on_safety(self, msg: String) -> None:
        packet = self._json_or_empty(msg.data)
        self.safety_status = str(packet.get("status", self.safety_status))

    def on_timer(self) -> None:
        self.get_logger().info("\n" + self._status_block() + "\n")

    def _status_block(self) -> str:
        selected = self.fusion_packet.get("selected") or {}
        mission = str(self.mission_state.get("state", "UNKNOWN"))
        active = str(self.mission_state.get("active_target", "-"))
        released = list(self.mission_state.get("released_targets", []) or [])
        search_target = self.mission_state.get("search_target") or ["-", "-"]
        search_index = int(self.mission_state.get("search_index", 0) or 0)
        lane = (search_index // 2) + 1

        opencv_status = "DETECTED" if self.raw_packet.get("opencv") else "WAIT"
        yolo_verified = bool(selected.get("yolo_verified", False))
        yolo_ran = bool(self.raw_packet.get("yolo_ran", False))
        yolo_status = "VERIFIED" if yolo_verified else ("CHECKED" if yolo_ran else "WAIT")
        iou_value = float(selected.get("yolo_iou", 0.0)) if selected else 0.0
        iou_status = "MATCHED" if yolo_verified else ("NO_MATCH" if yolo_ran else "WAIT")

        fusion_target = str(selected.get("target_type", "-")) if selected else "-"
        fusion_state = str(selected.get("target_state", self.fusion_packet.get("state", "SEARCH")))
        fusion_conf = float(selected.get("fusion_confidence", 0.0)) if selected else 0.0

        error = selected.get("error") or [None, None]
        err_x = self._fmt_number(error[0])
        err_y = self._fmt_number(error[1])
        center_status = self._center_status(error)

        release_gate = bool(selected.get("release_gate", selected.get("drop_ready", False)))
        lock_counter = int(selected.get("lock_counter", 0)) if selected else 0
        payload_target = self._payload_target(active)
        resume_state, next_target = self._resume_state(active, released, mission)

        return "\n".join(
            [
                f"[MISSION] state={mission}  geofence={self.safety_status}  lane={lane:02d}",
                f"[SEARCH ] next_point=({search_target[0]},{search_target[1]})  target={active}",
                f"[VISION ] opencv={opencv_status}  yolo={yolo_status}  iou={iou_status}({iou_value:.2f})",
                f"[FUSION ] target={fusion_target}  state={fusion_state}  conf={fusion_conf:.2f}",
                f"[ALIGN  ] error_x={err_x}  error_y={err_y}  center={center_status}",
                f"[GATE   ] release_gate={str(release_gate).upper()}  lock_counter={lock_counter}/{config.LOCK_MIN_FRAMES}",
                f"[PAYLOAD] released={len(released)}/2  target={payload_target}",
                f"[MISSION] state={resume_state}  next_target={next_target}",
            ]
        )

    def _payload_target(self, active: str) -> str:
        if self.last_event.get("command") == "drop_payload":
            return str(self.last_event.get("target_type", active))
        return active if self.mission_state.get("released_targets") else "-"

    @staticmethod
    def _fmt_number(value) -> str:
        if value is None or value == "-":
            return "-"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if abs(numeric - round(numeric)) < 1e-6:
            return str(int(round(numeric)))
        return f"{numeric:.1f}"

    @staticmethod
    def _center_status(error) -> str:
        if not error or error[0] is None or error[1] is None:
            return "WAIT"
        try:
            err_x = float(error[0])
            err_y = float(error[1])
        except (TypeError, ValueError):
            return "WAIT"
        return "OK" if abs(err_x) <= config.CENTER_TOL_X and abs(err_y) <= config.CENTER_TOL_Y else "ADJUST"

    @staticmethod
    def _resume_state(active: str, released: list, mission: str) -> tuple[str, str]:
        if mission == "MISSION_COMPLETE":
            return "MISSION_COMPLETE", "-"
        if mission == "RETURN_HOME":
            return "RTL", "-"
        if mission == "POST_DROP_HOVER":
            return "CLIMB", active
        if released and mission == "SEARCH_TARGET":
            return "RESUME_SEARCH", active
        return mission, active

    def _mission_line(self) -> str:
        state = self.mission_state.get("state", "UNKNOWN")
        active = self.mission_state.get("active_target", "-")
        released = ",".join(self.mission_state.get("released_targets", []) or []) or "-"
        nav_x = float(self.mission_state.get("nav_x", 0.0))
        nav_y = float(self.mission_state.get("nav_y", 0.0))
        alt = float(self.mission_state.get("altitude_m", 0.0))
        search_index = self.mission_state.get("search_index", "-")
        search_axis = self.mission_state.get("search_axis", "-")
        search_target = self.mission_state.get("search_target") or ["-", "-"]
        search_start = self.mission_state.get("search_start", "-")
        return (
            f"MISSION state={state} active={active} released={released} "
            f"nav=({nav_x:.1f},{nav_y:.1f}) alt={alt:.1f}m "
            f"search_start={search_start} search={search_index}:{search_axis}->({search_target[0]},{search_target[1]})"
        )

    def _stateflow_line(self) -> str:
        mission = str(self.mission_state.get("state", "UNKNOWN"))
        selected = self.fusion_packet.get("selected") or {}
        fusion_state = str(selected.get("target_state", self.fusion_packet.get("state", "SEARCH")))
        released = list(self.mission_state.get("released_targets", []) or [])
        active = self.mission_state.get("active_target", "-")
        opencv_seen = bool(self.raw_packet.get("opencv", []))
        yolo_seen = bool(self.raw_packet.get("yolo", []))
        yolo_ran = bool(self.raw_packet.get("yolo_ran", False))
        yolo_verified = bool(selected.get("yolo_verified", False))
        release_gate = bool(selected.get("release_gate", selected.get("drop_ready", False)))
        lock_counter = int(selected.get("lock_counter", 0)) if selected else 0

        active_phase = self._active_phase(mission, fusion_state, opencv_seen, yolo_verified, release_gate)
        arm_takeoff = self._stage_status(
            mission,
            active={"SET_GUIDED", "ARM", "TAKEOFF"},
            done={
                "SEARCH_TARGET",
                "TARGET_CANDIDATE",
                "TARGET_ALIGN",
                "TARGET_VERIFY",
                "DROP_TARGET",
                "POST_DROP_HOVER",
                "RETURN_HOME",
                "LAND",
                "MISSION_COMPLETE",
            },
        )
        mission_start = "OK" if mission not in {"UNKNOWN", "INIT", "WAIT_START", "WAIT_FOR_CAMERA"} else "WAIT"
        waypoint = "RUN" if mission == "SEARCH_TARGET" else ("OK" if released else "WAIT")
        search_area = "RUN" if mission == "SEARCH_TARGET" and not opencv_seen else ("OK" if opencv_seen else "WAIT")
        opencv = "DETECTED" if opencv_seen else "WAIT"
        yolo = "VERIFIED" if yolo_verified else ("RUN/NEGATIVE" if yolo_ran else ("SEEN" if yolo_seen else "SKIP"))
        fusion = fusion_state if selected else "SEARCH"
        alignment = self._stage_status(mission, active={"TARGET_CANDIDATE", "TARGET_ALIGN", "TARGET_VERIFY"}, done={"DROP_TARGET", "POST_DROP_HOVER", "RETURN_HOME", "MISSION_COMPLETE"})
        release = "OPEN" if release_gate else "CLOSED"
        payload = f"{len(released)}/2"
        second_payload = "OK" if len(released) >= 2 else ("RUN" if len(released) == 1 else "WAIT")
        climb = "RUN" if mission == "POST_DROP_HOVER" else ("OK" if released and mission in {"SEARCH_TARGET", "RETURN_HOME", "MISSION_COMPLETE"} else "WAIT")
        complete = "OK" if mission == "MISSION_COMPLETE" else "WAIT"
        rtl_land = "RUN" if mission in {"RETURN_HOME", "LAND"} else ("OK" if mission == "MISSION_COMPLETE" else "WAIT")

        return (
            f"STATEFLOW active={active_phase} target={active} | "
            f"ARM/TAKEOFF={arm_takeoff} MISSION_START={mission_start} NAV={waypoint} SEARCH={search_area} "
            f"OpenCV={opencv} YOLO={yolo} FUSION={fusion} ALIGN={alignment} "
            f"RELEASE_GATE={release} lock={lock_counter} PAYLOAD={payload} "
            f"CLIMB={climb} SECOND_PAYLOAD={second_payload} COMPLETE={complete} RTL/LAND={rtl_land}"
        )

    def _perception_line(self) -> str:
        opencv = list(self.raw_packet.get("opencv", []) or [])
        yolo = list(self.raw_packet.get("yolo", []) or [])
        cv_summary = self._detection_summary(opencv)
        yolo_summary = self._detection_summary(yolo)
        frame_id = self.raw_packet.get("frame_id", "-")
        yolo_error = self.raw_packet.get("yolo_error")
        yolo_ran = bool(self.raw_packet.get("yolo_ran", False))
        yolo_age = self.raw_packet.get("yolo_age_frames", "-")
        yolo_status = "fresh" if yolo_ran else "skipped/no fresh result"
        suffix = f" yolo_error={yolo_error}" if yolo_error else ""
        return (
            f"FRAME #{frame_id} OpenCV={cv_summary} YOLO={yolo_summary} "
            f"YOLO_status={yolo_status} age={yolo_age}{suffix}"
        )

    def _fusion_line(self) -> str:
        selected = self.fusion_packet.get("selected") or {}
        if not selected:
            state = self.fusion_packet.get("state", "SEARCH")
            return f"TRACK state={state} selected=- hedef aranıyor targets={len(self.fusion_packet.get('targets', []) or [])}"
        target = selected.get("target_type", "-")
        state = selected.get("target_state", selected.get("state", "-"))
        det_conf = float(selected.get("confidence", 0.0))
        fus_conf = float(selected.get("fusion_confidence", 0.0))
        error = selected.get("error") or ["-", "-"]
        yolo_verified = bool(selected.get("yolo_verified", False))
        release_gate = bool(selected.get("release_gate", selected.get("drop_ready", False)))
        center = selected.get("center") or ["-", "-"]
        yolo_iou = float(selected.get("yolo_iou", 0.0))
        lock_counter = int(selected.get("lock_counter", 0))
        unstable_counter = int(selected.get("unstable_counter", 0))
        message = self._tracking_message(str(state), yolo_verified, release_gate)
        return (
            f"TRACK target={target} state={state} {message} det={det_conf:.2f} fus={fus_conf:.2f} "
            f"err=({error[0]},{error[1]}) center=({center[0]},{center[1]}) "
            f"yolo_verified={yolo_verified} iou={yolo_iou:.2f} "
            f"lock_frames={lock_counter} unstable_frames={unstable_counter} release_gate={release_gate}"
        )

    def _drone_line(self) -> str:
        return (
            f"DRONE connected={self.drone_state.connected} armed={self.drone_state.armed} "
            f"mode={self.drone_state.mode} local=({self.local_position.x:.1f},{self.local_position.y:.1f},{self.local_position.z:.1f})"
        )

    def _event_line(self) -> str:
        if not self.last_event:
            return ""
        command = self.last_event.get("command", "-")
        target = self.last_event.get("target_type", "-")
        dry_run = bool(self.last_event.get("dry_run", False))
        servo = self.last_event.get("servo", "-")
        estimate = self.last_event.get("drop_estimate", {}) or {}
        prefix = "PAYLOAD_RELEASED" if command == "drop_payload" else f"EVENT command={command}"
        if estimate:
            err = float(estimate.get("distance_to_center_m", 0.0))
            inside = bool(estimate.get("inside_target_footprint", False))
            return (
                f"{prefix} target={target} dry_run={dry_run} servo={servo} "
                f"drop_error={err:.2f}m inside={inside}"
            )
        return f"{prefix} target={target} dry_run={dry_run} servo={servo}"

    @staticmethod
    def _active_phase(
        mission: str,
        fusion_state: str,
        opencv_seen: bool,
        yolo_verified: bool,
        release_gate: bool,
    ) -> str:
        if mission in {"SET_GUIDED", "ARM", "TAKEOFF"}:
            return "ARM / TAKEOFF"
        if mission == "SEARCH_TARGET":
            if not opencv_seen:
                return "WAYPOINT NAVIGATION / SEARCH AREA"
            if not yolo_verified:
                return "OpenCV TARGET DETECTION -> YOLO VERIFICATION"
            if fusion_state == "LOCKED":
                return "FUSION LOCK"
            return "TARGET TRACKING"
        if mission in {"TARGET_CANDIDATE", "TARGET_ALIGN"}:
            return "TARGET ALIGNMENT"
        if mission == "TARGET_VERIFY":
            return "RELEASE GATE" if release_gate else "FUSION LOCK / TARGET VERIFY"
        if mission == "DROP_TARGET":
            return "PAYLOAD RELEASE"
        if mission == "POST_DROP_HOVER":
            return "CLIMB / NEXT TARGET"
        if mission == "RETURN_HOME":
            return "RTL / LAND"
        if mission == "MISSION_COMPLETE":
            return "MISSION COMPLETE"
        if mission in {"WAIT_FOR_CAMERA", "CONNECT_MAVLINK", "INIT", "WAIT_START"}:
            return "MISSION START"
        return mission

    @staticmethod
    def _stage_status(mission: str, active: set[str], done: set[str]) -> str:
        if mission in active:
            return "RUN"
        if mission in done:
            return "OK"
        return "WAIT"

    @staticmethod
    def _tracking_message(state: str, yolo_verified: bool, release_gate: bool) -> str:
        if release_gate:
            return "BIRAKMA_IZNI_ACIK: hedef merkezde ve kilit yeterli"
        if state == "LOCKED":
            return "KILITLI: hedef doğrulandı, merkez/lock süresi izleniyor"
        if state == "TRACKING":
            return "TAKIPTE: hedef görülüyor, bırakma için doğrulama bekleniyor"
        if state == "CANDIDATE":
            return "ADAY: yeni hedef, warm-up/doğrulama bekleniyor"
        if state == "UNSTABLE":
            return "KARARSIZ: OpenCV görüyor ama YOLO doğrulaması zayıf"
        if yolo_verified:
            return "YOLO doğruladı"
        return "hedef aranıyor/doğrulama bekleniyor"

    @staticmethod
    def _detection_summary(detections: list[dict]) -> str:
        if not detections:
            return "-"
        counts = Counter(str(det.get("target_type", "unknown")) for det in detections)
        best_by_type: dict[str, float] = {}
        for det in detections:
            target = str(det.get("target_type", "unknown"))
            best_by_type[target] = max(best_by_type.get(target, 0.0), float(det.get("confidence", 0.0)))
        return ",".join(f"{target}:{counts[target]}@{best_by_type[target]:.2f}" for target in sorted(counts))

    @staticmethod
    def _json_or_empty(text: str) -> dict:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def main() -> None:
    rclpy.init()
    node = MissionConsoleNode()
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
