cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 17)
project(sot)

set(OpenCV_DIR "/home/fbh/opencv-4.1.2/build")
set(Eigen3_DIR "/home/fbh/eigen-3.3.7/build")
set(cv_bridge_DIR "/home/fbh/Cv_Bridge/install/share/cv_bridge/cmake")

find_package(OpenCV 4.1.2 REQUIRED)
find_package(realsense2 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  rospy  
  roscpp
  rosbag
  OpenCV
  Eigen3
  cv_bridge
  image_transport
  message_filters
  std_msgs
  nav_msgs
  sensor_msgs
  darknet_ros_msgs  
)

catkin_package(

)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${realsense2_LIBRARY}
)

add_executable(track 
  src/msot_gentrk.cpp
  src/Hungarian.cpp
)

target_link_libraries(track ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${cv_bridge_LIBRARIES})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "cv_bridge version: ${cv_bridge_VERSION}")
