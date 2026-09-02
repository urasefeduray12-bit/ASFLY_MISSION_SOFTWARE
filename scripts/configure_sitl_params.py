#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from pymavlink import mavutil


CRITICAL_PARAMS = {
    "FRAME_CLASS": 1,
    "FRAME_TYPE": 1,
    "SERVO1_FUNCTION": 33,
    "SERVO2_FUNCTION": 34,
    "SERVO3_FUNCTION": 35,
    "SERVO4_FUNCTION": 36,
    "MOT_PWM_MIN": 1100,
    "MOT_PWM_MAX": 1900,
    "MNT1_TYPE": 0,
    "PLND_ENABLED": 0,
    "FS_CRASH_CHECK": 0,
    "ARMING_SKIPCHK": 65535,
}


def set_param(master, name: str, value: float, timeout_s: float) -> float:
    master.param_set_send(name, value)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is None:
            continue
        if msg.param_id.strip("\x00") == name:
            return float(msg.param_value)
    raise TimeoutError(f"Timed out while setting {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set critical ArduPilot SITL params for Teknofest Gazebo tests.")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14551")
    parser.add_argument("--source-system", type=int, default=245)
    parser.add_argument("--heartbeat-timeout", type=float, default=10.0)
    parser.add_argument("--param-timeout", type=float, default=3.0)
    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.connection, source_system=args.source_system)
    heartbeat = master.wait_heartbeat(timeout=args.heartbeat_timeout)
    if heartbeat is None:
        raise TimeoutError(f"No MAVLink heartbeat on {args.connection}")

    print(f"connected target={master.target_system}:{master.target_component}")
    for name, expected in CRITICAL_PARAMS.items():
        actual = set_param(master, name, expected, args.param_timeout)
        ok = abs(actual - float(expected)) < 0.5
        marker = "OK" if ok else "MISMATCH"
        print(f"{marker} {name}={actual:g} expected={expected:g}")
        if not ok:
            raise RuntimeError(f"{name} expected {expected}, got {actual}")


if __name__ == "__main__":
    main()
