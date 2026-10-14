cmake_minimum_required(VERSION 3.5)
project(magician_body_pose_estimation)

find_package(ament_cmake REQUIRED)
find_package(skeleton_msgs REQUIRED)

ament_export_dependencies(skeleton_msgs)
ament_package()
