#!/usr/bin/env bash
set -euo pipefail

# MAVProxy router for field use.
#
# Examples:
#   SITL:
#     MAVLINK_MASTER=tcp:127.0.0.1:5760 ./scripts/start_mavlink_router.sh
#
#   Pixhawk over USB:
#     MAVLINK_MASTER=/dev/ttyACM0,115200 ./scripts/start_mavlink_router.sh
#
#   Telemetry radio:
#     MAVLINK_MASTER=/dev/ttyUSB0,57600 ./scripts/start_mavlink_router.sh
#
# Outputs:
#   127.0.0.1:14550 -> ground-control station, e.g. QGroundControl or Mission Planner
#   127.0.0.1:14551 -> ROS 2 mavlink_bridge_node

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.ros_venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.ros_venv/bin/activate"
fi

export PATH="${PATH}:${HOME}/.local/bin"

MAVLINK_MASTER="${MAVLINK_MASTER:-tcp:127.0.0.1:5760}"
GCS_OUT="${GCS_OUT:-127.0.0.1:14550}"
ROS_OUT="${ROS_OUT:-127.0.0.1:14551}"

echo "[MAVLINK] master=${MAVLINK_MASTER}"
echo "[MAVLINK] gcs_out=${GCS_OUT}"
echo "[MAVLINK] ros_out=${ROS_OUT}"
echo "[MAVLINK] ROS config: config/mavlink_gcs_router.yaml"

exec mavproxy.py \
  --master="${MAVLINK_MASTER}" \
  --out="${GCS_OUT}" \
  --out="${ROS_OUT}"
