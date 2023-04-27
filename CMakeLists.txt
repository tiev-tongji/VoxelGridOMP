cmake_minimum_required(VERSION 3.20)
project(pcl_test)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS}  ${OpenMP_CXX_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(pcl_test main.cpp  voxel_grid_omp.cpp)
target_compile_options(pcl_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(pcl_test ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS})
