cmake_minimum_required(VERSION 2.8.3)
project(x1smsm)

#https://stackoverflow.com/a/38511897
SET(CMAKE_CXX_COMPILER /usr/bin/g++)
add_compile_options(-std=c++11 -Ofast)

find_package(PkgConfig)

find_package(CGAL REQUIRED)

INCLUDE_DIRECTORIES(include
  /usr/include/eigen3/)


# smsm ########################################################################
add_library(smsm_lib
  src/smsm.cpp)

target_link_libraries(smsm_lib
  fftw3
  CGAL::CGAL_Core)

# The executable ###############################################################
add_executable(smsm_node
  src/smsm_node.cpp)

target_link_libraries(smsm_node
  smsm_lib)
