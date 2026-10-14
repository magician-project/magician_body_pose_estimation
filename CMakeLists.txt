cmake_minimum_required(VERSION 3.5)
project(magician_body_pose_estimation)

find_package(ament_cmake REQUIRED)


#AMMAR
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/Joint3D.msg" "msg/Skeleton.msg" "msg/Skeletons.msg" )

ament_export_dependencies(rosidl_default_runtime)

#ALTINAY
#find_package(skeleton_msgs REQUIRED)
#ament_export_dependencies(skeleton_msgs)


ament_package()
