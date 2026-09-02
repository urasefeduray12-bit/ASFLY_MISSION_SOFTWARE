#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

from pymavlink import mavutil


def recv_recent(master, seconds: float = 1.0) -> dict[str, object]:
    deadline = time.time() + seconds
    latest: dict[str, object] = {}
    while time.time() < deadline:
        msg = master.recv_match(blocking=True, timeout=0.2)
        if msg is None:
            continue
        msg_type = msg.get_type()
        latest[msg_type] = msg
        if msg_type == "STATUSTEXT":
            print(f"STATUSTEXT {msg.text}")
        elif msg_type == "COMMAND_ACK":
            print(f"ACK command={msg.command} result={msg.result}")
    return latest


def set_mode(master, mode: str, timeout_s: float = 5.0) -> None:
    mapping = master.mode_mapping()
    mode_id = mapping.get(mode)
    if mode_id is None:
        raise RuntimeError(f"Mode {mode!r} not available: {sorted(mapping)}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        latest = recv_recent(master, 0.3)
        heartbeat = latest.get("HEARTBEAT")
        if heartbeat is not None and mavutil.mode_string_v10(heartbeat) == mode:
            return
    raise TimeoutError(f"Timed out waiting for mode {mode}")


def arm(master, timeout_s: float = 8.0) -> None:
    master.arducopter_arm()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        latest = recv_recent(master, 0.3)
        heartbeat = latest.get("HEARTBEAT")
        if heartbeat is not None and heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            return
    raise TimeoutError("Timed out waiting for armed state")


def takeoff(master, altitude_m: float) -> None:
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        altitude_m,
    )


def print_sample(latest: dict[str, object]) -> tuple[float, float]:
    local = latest.get("LOCAL_POSITION_NED")
    attitude = latest.get("ATTITUDE")
    heartbeat = latest.get("HEARTBEAT")
    altitude_m = 0.0
    roll_abs = 0.0
    fields: list[str] = []
    if local is not None:
        altitude_m = max(0.0, -float(local.z))
        fields.append(f"alt={altitude_m:.2f} pos=({local.x:.2f},{local.y:.2f},{local.z:.2f})")
    if attitude is not None:
        roll_abs = abs(float(attitude.roll))
        fields.append(f"roll={attitude.roll:.2f} pitch={attitude.pitch:.2f}")
    if heartbeat is not None:
        armed = bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        fields.append(f"mode={mavutil.mode_string_v10(heartbeat)} armed={armed}")
    print(" ".join(fields) if fields else "no telemetry sample")
    return altitude_m, roll_abs


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ArduPilot/Gazebo takeoff stability smoke test.")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14551")
    parser.add_argument("--source-system", type=int, default=240)
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--observe-seconds", type=float, default=12.0)
    parser.add_argument("--max-roll-rad", type=float, default=0.8)
    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.connection, source_system=args.source_system)
    if master.wait_heartbeat(timeout=10.0) is None:
        raise TimeoutError(f"No MAVLink heartbeat on {args.connection}")

    print(f"connected target={master.target_system}:{master.target_component}")
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, args.altitude)

    max_altitude = 0.0
    max_roll = 0.0
    deadline = time.time() + args.observe_seconds
    while time.time() < deadline:
        altitude_m, roll_abs = print_sample(recv_recent(master, 1.0))
        max_altitude = max(max_altitude, altitude_m)
        max_roll = max(max_roll, roll_abs)

    print(f"summary max_alt={max_altitude:.2f} max_roll_rad={max_roll:.2f}")
    try:
        set_mode(master, "LAND", timeout_s=2.0)
    except Exception as exc:
        print(f"land_mode_error={exc}")

    if max_altitude < args.altitude * 0.5:
        raise RuntimeError("Takeoff did not reach enough altitude.")
    if not math.isfinite(max_roll) or max_roll > args.max_roll_rad:
        raise RuntimeError("Vehicle roll exceeded stability threshold.")


if __name__ == "__main__":
    main()
