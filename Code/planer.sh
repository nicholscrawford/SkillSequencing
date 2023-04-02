#!/bin/bash

roslaunch moveit_interface iiwa_reflex_moveit_interface_service.launch use_rviz:=True &

python ../../isaacgym_publisher.py --fake &

python gym_planner.py