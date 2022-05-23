#!/bin/bash

export TURTLEBOT3_MODEL=burger

Xvfb :1 -screen 0 1600x1200x16  &


ROS_MASTER_URI="0.0.0.0"
source /opt/ros/melodic/setup.bash
source /content/mapless/catkin_ws/devel/setup.bash

DISPLAY=:1.0 roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
