#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/omer/Downloads/yolosingleclass/asfly_singleclass_target_square_package}"
ARDUPILOT_DIR="${ARDUPILOT_DIR:-/home/omer/ardupilot}"
ARDUPILOT_VENV="${ARDUPILOT_VENV:-/home/omer/venv-ardupilot}"
GAZEBO_WORLD="${GAZEBO_WORLD:-/home/omer/ardupilot_gazebo/worlds/teknofest_bozkir.sdf}"
ARDUPILOT_GAZEBO_DIR="${ARDUPILOT_GAZEBO_DIR:-/home/omer/ardupilot_gazebo}"
LOG_DIR="${LOG_DIR:-/tmp/teknofest_logs}"
ROS_VENV_SITE="${PROJECT_DIR}/.ros_venv/lib/python3.12/site-packages"

mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/matplotlib"

assert_clean_start() {
  local existing
  existing="$(
    {
      pgrep -a -f "gz sim .*teknofest_bozkir.sdf" || true
      pgrep -a -f "gz sim -g" || true
      pgrep -a -f "arducopter --model JSON" || true
      pgrep -a -f "ros2 launch teknofest_iha obs_recording.launch.py" || true
      pgrep -a -f "ros_gz_bridge.*parameter_bridge" || true
    } | sed '/^[[:space:]]*$/d'
  )"
  if [ -n "${existing}" ]; then
    cat <<EOF
Existing Teknofest/Gazebo processes are already running:
${existing}

Starting another stack can make Gazebo spin forever or crash.
Stop the old stack first:
  ${PROJECT_DIR}/scripts/stop_teknofest_stack.sh

If you intentionally want to start anyway:
  TEKNOFEST_ALLOW_EXISTING=1 ${PROJECT_DIR}/scripts/prepare_obs_recording.sh
EOF
    if [ "${TEKNOFEST_ALLOW_EXISTING:-0}" != "1" ]; then
      exit 1
    fi
  fi
}

assert_clean_start

open_terminal() {
  local title="$1"
  local command="$2"
  local wrapped
  wrapped="${command}; status=\$?; echo; echo '[${title}] process exited with status '\${status}; exec bash"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="${title}" -- bash -lc "${wrapped}"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "${title}" -e bash -lc "${wrapped}" &
  else
    echo "No supported terminal found. Install gnome-terminal or xterm." >&2
    return 1
  fi
}

start_debug_viewer() {
  nohup bash -lc "
    cd '${PROJECT_DIR}'
    source_setup() { set +u; source \"\$1\"; set -u; }
    export ROS_LOG_DIR='${LOG_DIR}/ros'
    export MPLCONFIGDIR='${LOG_DIR}/matplotlib'
    export PYTHONPATH='${ROS_VENV_SITE}:'\${PYTHONPATH:-}
    source .ros_venv/bin/activate
    source_setup /opt/ros/jazzy/setup.bash
    source_setup install/setup.bash
    ros2 run teknofest_iha debug_viewer_node
  " > "${LOG_DIR}/debug_viewer.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/debug_viewer.pid"
}

open_terminal "TEKNOFEST_GAZEBO_SERVER" "
  export GZ_SIM_SYSTEM_PLUGIN_PATH='${ARDUPILOT_GAZEBO_DIR}/build:'\${GZ_SIM_SYSTEM_PLUGIN_PATH:-}
  export GZ_SIM_RESOURCE_PATH='${ARDUPILOT_GAZEBO_DIR}/models:${ARDUPILOT_GAZEBO_DIR}/worlds:'\${GZ_SIM_RESOURCE_PATH:-}
  gz sim -s -v4 -r '${GAZEBO_WORLD}'
"

sleep 6

open_terminal "TEKNOFEST_GAZEBO_GUI" "
  export GZ_SIM_RESOURCE_PATH='${ARDUPILOT_GAZEBO_DIR}/models:${ARDUPILOT_GAZEBO_DIR}/worlds:'\${GZ_SIM_RESOURCE_PATH:-}
  gz sim -g
"

sleep 2

open_terminal "TEKNOFEST_ARDUPILOT" "
  cd '${ARDUPILOT_DIR}'
  if [ -f '${ARDUPILOT_VENV}/bin/activate' ]; then source '${ARDUPILOT_VENV}/bin/activate'; fi
  export PATH=\$PATH:\$HOME/.local/bin
  ./build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 --sim-address=127.0.0.1 -I0
"

sleep 5

open_terminal "TEKNOFEST_ROS" "
  cd '${PROJECT_DIR}'
  source_setup() { set +u; source \"\$1\"; set -u; }
  export ROS_LOG_DIR='${LOG_DIR}/ros'
  export MPLCONFIGDIR='${LOG_DIR}/matplotlib'
  export PYTHONPATH='${ROS_VENV_SITE}:'\${PYTHONPATH:-}
  source .ros_venv/bin/activate
  source_setup /opt/ros/jazzy/setup.bash
  source_setup install/setup.bash
  ros2 launch teknofest_iha obs_recording.launch.py autostart:=false record_output_dir:=/tmp record_max_duration_s:=300.0
"

sleep 5
start_debug_viewer

cat <<EOF

OBS preparation started.

Capture these windows in OBS:
  1. Gazebo overview:       Gazebo Sim
  2. UAV camera overlay:    teknofest_iha_debug
  3. ROS terminal:          TEKNOFEST_ROS
  4. ArduPilot terminal:    TEKNOFEST_ARDUPILOT

Mission manager is running in WAIT_START mode.
Start OBS recording first, then run:
  ${PROJECT_DIR}/scripts/start_mission_for_obs.sh

Debug viewer log:
  ${LOG_DIR}/debug_viewer.log

EOF
