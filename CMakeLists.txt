cmake_minimum_required(VERSION 3.2)

project(position-based-fluids)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_GLFW "Build target igl::glfw" ON)
include(libigl)

#include directories
#include_directories(${PROJECT_SOURCE_DIR}/extern/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${LIBIGL_INCLUDE_DIR})

#include source code
file(GLOB H_FILES ${PROJECT_SOURCE_DIR}/include/*.h)
#file(GLOB H_SHARED_FILES ${PROJECT_SOURCE_DIR}/shared/include/*.h)
file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
#file(GLOB CPP_SHARED_FILES ${PROJECT_SOURCE_DIR}/shared/src/*.cpp)

#build executable 
add_executable(${PROJECT_NAME} main.cpp ${H_FILES} ${CPP_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)