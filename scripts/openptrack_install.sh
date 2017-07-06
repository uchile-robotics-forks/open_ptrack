#!/bin/bash

# Clone OpenPTrack into ROS workspace:
cd ~/catkin_ws/src
git clone https://github.com/OpenPTrack/open_ptrack.git
cd ~/catkin_ws/src/open_ptrack/scripts
chmod +x *.sh

echo Clon
# Calibration_toolkit installation:
./calibration_toolkit_install.sh
echo Calibration toolkit
# Update libfreenect driver for Kinect:
./libfreenect_update.sh
echo Libfreenect
# Install SwissRanger driver:
./mesa_install.sh
echo SwissRanger
# Building everything
cd ~/catkin_ws
catkin_make --pkg calibration_msgs
catkin_make --pkg opt_msgs
catkin_make --force-cmake
echo Build
