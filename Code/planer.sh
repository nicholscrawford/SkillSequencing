#!/bin/bash

gnome-terminal --tab --title="moveit and rviz" -- roslaunch moveit_interface iiwa_reflex_moveit_interface_service.launch use_rviz:=True
gnome-terminal --tab --title="gym publisher" -- python ../../isaacgym_publisher.py --fake 
#python gym_planner.py
