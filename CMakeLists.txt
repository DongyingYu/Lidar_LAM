cmake_minimum_required(VERSION 2.8)
project(Lidar_LAM)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msg
    pcl_conversion
    pcl_ros
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msg message_runtime
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

catkin_package(
CATKIN_DEPENDS geomrtry_msgs nav_msgs roscpp rospy std_mags
DEPENDS EIGEN3 PCL OpenCV
INCLUDE_DIRS
)

add_executable(lidar_lam ...)
target_link_libraries(lidar_lam ${catkin_LIBRARIES} ${PCL_LIBRARIES})