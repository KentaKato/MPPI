cmake_minimum_required(VERSION 3.10)

project(MPPI CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(main
  src/main.cpp
)

target_link_libraries(main
  ${PROJECT_NAME}
)


