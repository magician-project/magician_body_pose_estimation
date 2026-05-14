#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source ~/.bashrc
source src/magician_body_pose_estimation/D-PoSE/venv/bin/activate
source install/setup.bash 

cd src/magician_body_pose_estimation
python3 magician_body_pose_estimation.py $@


exit 0
