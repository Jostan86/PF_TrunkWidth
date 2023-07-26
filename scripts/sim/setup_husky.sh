#!/bin/bash

export HUSKY_REALSENSE_ENABLED="1"
export HUSKY_REALSENSE_TOPIC="realsense"
export HUSKY_REALSENSE_PREFIX="camera"
export HUSKY_REALSENSE_PARENT="top_plate_link"
export HUSKY_REALSENSE_XYZ="0 -0.25 0.2"
export HUSKY_REALSENSE_RPY="0 0 -1.5707963"


export HUSKY_LMS1XX_ENABLED="1"
export HUSKY_LMS1XX_TOPIC="front/scan"
export HUSKY_LMS1XX_TOWER="1"
export HUSKY_LMS1XX_PREFIX="front"
export HUSKY_LMS1XX_PARENT="top_plate_link"
export HUSKY_LMS1XX_XYZ="0.2206 0.0 0.00635"
export HUSKY_LMS1XX_RPY="0.0 0.0 0.0"

#roslaunch sim_pf_trunk_width run_sim.launch
#roslaunch sim_pf_trunk_width run_UFO_sim.launch
roslaunch sim_pf_trunk_width run_pear_sim.launch
