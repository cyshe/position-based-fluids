cmake_minimum_required(VERSION 3.2)

project(position-based-fluids)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include)

#include source code
file(GLOB H_FILES ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Polyscope 
add_subdirectory("deps/polyscope")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#build executable 
add_executable(${PROJECT_NAME} main.cpp ${CPP_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC polyscope Eigen3::Eigen)