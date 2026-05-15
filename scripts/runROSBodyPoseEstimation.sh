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
python3 magician_body_pose_estimation.py --use-aruco --display --render --insist-camera --input /dev/video4 $@


exit 0
