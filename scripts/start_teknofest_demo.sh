#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package"
ARDUPILOT_DIR="/home/omer/ardupilot"
GAZEBO_WORLD="/home/omer/ardupilot_gazebo/worlds/teknofest_bozkir.sdf"
LOG_DIR="/tmp/teknofest_logs"
ROS_VENV_SITE="${PROJECT_DIR}/.ros_venv/lib/python3.12/site-packages"

mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/matplotlib"

export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/omer/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="/home/omer/ardupilot_gazebo/models:/home/omer/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"

start_detached() {
  local name="$1"
  shift
  echo "Starting ${name}..."
  nohup "$@" > "${LOG_DIR}/${name}.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/${name}.pid"
  echo "  pid=$(cat "${LOG_DIR}/${name}.pid") log=${LOG_DIR}/${name}.log"
}

start_detached_shell() {
  local name="$1"
  local command="$2"
  echo "Starting ${name}..."
  nohup bash -lc "${command}" > "${LOG_DIR}/${name}.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/${name}.pid"
  echo "  pid=$(cat "${LOG_DIR}/${name}.pid") log=${LOG_DIR}/${name}.log"
}

start_detached "gazebo" gz sim -v4 -r "${GAZEBO_WORLD}"
sleep 6

start_detached_shell "ardupilot_sitl" \
  "cd '${ARDUPILOT_DIR}' && ./build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 --sim-address=127.0.0.1 -I0"
sleep 8

start_detached_shell "ros_full_stack" \
  "cd '${PROJECT_DIR}' && source_setup() { set +u; source \"\$1\"; set -u; } && export MPLCONFIGDIR='${LOG_DIR}/matplotlib' && export PYTHONPATH='${ROS_VENV_SITE}:'\${PYTHONPATH:-} && source .ros_venv/bin/activate && source_setup /opt/ros/jazzy/setup.bash && source_setup install/setup.bash && ros2 launch teknofest_iha full_sim.launch.py"
sleep 5

start_detached_shell "mission_recorder" \
  "cd '${PROJECT_DIR}' && source_setup() { set +u; source \"\$1\"; set -u; } && export MPLCONFIGDIR='${LOG_DIR}/matplotlib' && export PYTHONPATH='${ROS_VENV_SITE}:'\${PYTHONPATH:-} && source .ros_venv/bin/activate && source_setup /opt/ros/jazzy/setup.bash && source_setup install/setup.bash && ros2 launch teknofest_iha record_mission.launch.py overlay_mode:=fusion output_dir:=/tmp"

echo
echo "Teknofest demo stack started."
echo "Logs: ${LOG_DIR}"
echo "Recorder output: /tmp/teknofest_mission_*.mp4"
