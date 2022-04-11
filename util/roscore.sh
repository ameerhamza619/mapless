#!/bin/bash
export DISPLAY =: 1
ROS_MASTER_URI="0.0.0.0"
source /opt/ros/melodic/setup.bash
source /content/mapless/catkin_ws/devel/setup.bash
roslaunch turtlebot3_gazebo train_env2.launch