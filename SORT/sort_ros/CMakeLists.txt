cmake_minimum_required(VERSION 3.0.2)
project(sort_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy  
  roscpp
  std_msgs
  sensor_msgs
  darknet_ros_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros-yolo-sort
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  src/sortt.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
