#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source ~/.bashrc
source src/magician_body_pose_estimation/D-PoSE/venv/bin/activate
source install/setup.bash 

cd src/magician_body_pose_estimation

#Specific invokation for Pilot PC 
python3 magician_body_pose_estimation.py  --display --render --insist-camera --aruco-marker-id 20 --aruco-marker-length 0.19 --input /dev/video4 --width 1280 --height 720 --static-camera --aruco-xyz -0.17 -0.00 -0.20 --aruco-rpy 90 0 -90 --fx 924 --fy 926  --cx 640 --cy 360 "$@"


exit 0
