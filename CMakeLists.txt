cmake_minimum_required(VERSION 2.8)
project(llam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    pcl_conversions
    pcl_ros
)

find_package(PCL 1.8 REQUIRED)
find_package(Eigen3)
SET(OpenCV_DIR "/usr/local/opencv3410/share/OpenCV")
find_package(OpenCV REQUIRED)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

catkin_package(
CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
DEPENDS EIGEN3 PCL OpenCV
INCLUDE_DIRS
)

add_executable(lidar_feature 
    src/system_ros.cc
    src/feature_extract.cc
)
target_link_libraries(lidar_feature ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(lidar_lam 
    src/lam_ros.cc
    src/lam_process.h
    src/lam_process.cc
    src/third_lib.cc
    src/third_lib.h
    src/localization_and_mapping.cc
    src/utils.cc
    src/utils.h
    src/basic_data.cc
    src/basic_data.h
    src/parameters.cc
    src/parameters.h
)
target_link_libraries(lidar_lam ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})