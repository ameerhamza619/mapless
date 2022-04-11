#!/bin/bash

export TURTLEBOT3_MODEL=burger
ROS_MASTER_URI="0.0.0.0"
source /opt/ros/melodic/setup.bash
source /content/mapless/catkin_ws/devel/setup.bash
roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
