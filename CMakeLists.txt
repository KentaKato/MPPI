cmake_minimum_required(VERSION 3.10)

project(MPPI CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/cost_map.cpp
  src/obstacles.cpp
  src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

add_executable(main
  src/main.cpp
)

target_link_libraries(main
  ${PROJECT_NAME}
)

add_executable(test_cost_map
  test/test_cost_map.cpp
)

target_link_libraries(test_cost_map
  ${PROJECT_NAME}
)

