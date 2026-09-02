#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package}"
LOG_DIR="${LOG_DIR:-/tmp/teknofest_logs}"
ROS_VENV_SITE="${PROJECT_DIR}/.ros_venv/lib/python3.12/site-packages"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}/ros"
mkdir -p "${LOG_DIR}/matplotlib"
export ROS_LOG_DIR="${LOG_DIR}/ros"
export MPLCONFIGDIR="${LOG_DIR}/matplotlib"
export PYTHONPATH="${ROS_VENV_SITE}:${PYTHONPATH:-}"

source_setup() {
  set +u
  source "$1"
  set -u
}

source .ros_venv/bin/activate
source_setup /opt/ros/jazzy/setup.bash
source_setup install/setup.bash

echo "Sending mission start command..."
for _ in 1 2 3; do
  ros2 topic pub --once /mission/cmd_start std_msgs/msg/String "{data: start}" >/dev/null
  sleep 0.3
done

echo "Mission start command sent. Current mission state:"
ros2 topic echo /mission/state --once
