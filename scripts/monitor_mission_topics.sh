#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-/tmp/teknofest_logs}"
ROS_VENV_SITE="${PROJECT_DIR}/.ros_venv/lib/python3.12/site-packages"
mkdir -p "${LOG_DIR}/matplotlib"
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

exec ros2 run teknofest_iha mission_console_node
