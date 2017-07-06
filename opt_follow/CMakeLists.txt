cmake_minimum_required(VERSION 2.8.3)
project(opt_follow)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
find_package(catkin REQUIRED COMPONENTS cmake_modules roscpp rospy rosconsole image_transport cv_bridge std_msgs sensor_msgs opt_utils opt_msgs detection message_generation)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
add_definitions(${OpenCV_DEFINITIONS})

find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS} include ${catkin_INCLUDE_DIRS})


include_directories(cfg/cpp)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp rospy cv_bridge image_transport sensor_msgs std_msgs geometry_msgs opt_utils opt_msgs detection)

add_executable(follower apps/following.cpp src/FeatureExtractor.cpp)
target_link_libraries(follower ${catkin_LIBRARIES})

add_executable(depth_follower apps/following_depth.cpp src/FeatureExtractor.cpp)
#add_dependencies(depth_follower ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
target_link_libraries(depth_follower ${catkin_LIBRARIES})

add_executable(follower_client apps/following_client.cpp)
target_link_libraries(follower_client ${catkin_LIBRARIES})
add_dependencies(follower_client opt_msgs_gencpp)



