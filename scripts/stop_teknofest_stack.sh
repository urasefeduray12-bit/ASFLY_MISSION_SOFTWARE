#!/usr/bin/env bash
set -euo pipefail

patterns=(
  "gz sim .*teknofest_bozkir.sdf"
  "gz sim -g"
  "arducopter --model JSON"
  "ros2 launch teknofest_iha obs_recording.launch.py"
  "ros_gz_bridge.*parameter_bridge"
  "teknofest_iha.*debug_viewer_node"
  "teknofest_iha.*mission_console_node"
)

echo "Stopping Teknofest Gazebo / ROS / ArduPilot demo processes..."

for pattern in "${patterns[@]}"; do
  matches="$(pgrep -a -f "${pattern}" || true)"
  if [ -z "${matches}" ]; then
    continue
  fi
  echo
  echo "Pattern: ${pattern}"
  echo "${matches}"
  pkill -TERM -f "${pattern}" || true
done

sleep 2

for pattern in "${patterns[@]}"; do
  matches="$(pgrep -a -f "${pattern}" || true)"
  if [ -z "${matches}" ]; then
    continue
  fi
  echo
  echo "Force stopping remaining pattern: ${pattern}"
  echo "${matches}"
  pkill -KILL -f "${pattern}" || true
done

echo
echo "Remaining matching processes:"
{
  for pattern in "${patterns[@]}"; do
    pgrep -a -f "${pattern}" || true
  done
} | sed '/^[[:space:]]*$/d' || true

echo "Done."
