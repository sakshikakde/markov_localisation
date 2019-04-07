cmake_minimum_required(VERSION 2.8.3)
project(markov_localisation)


find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    )

catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES markov_localisation
    #  CATKIN_DEPENDS roscpp rospy std_msgs
    #  DEPENDS system_lib
    )

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    )


add_executable(${PROJECT_NAME}_node src/main.cpp src/bayesian_filter.cpp src/help_functions.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
