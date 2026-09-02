#!/usr/bin/env bash
set -euo pipefail

# Starts only the real/field ROS stack. It does not start Gazebo or SITL.
# Start MAVProxy first with scripts/start_mavlink_router.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f ".ros_venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".ros_venv/bin/activate"
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

if [[ ! -f "install/setup.bash" ]]; then
  echo "[ROS] install/setup.bash not found. Run: colcon build --symlink-install" >&2
  exit 1
fi

# shellcheck disable=SC1091
source install/setup.bash

AUTOSTART="${AUTOSTART:-false}"
CONSOLE_RATE_HZ="${CONSOLE_RATE_HZ:-5.0}"

echo "[ROS] launching field_mission.launch.py autostart=${AUTOSTART}"
exec ros2 launch teknofest_iha field_mission.launch.py \
  autostart:="${AUTOSTART}" \
  console_rate_hz:="${CONSOLE_RATE_HZ}"
