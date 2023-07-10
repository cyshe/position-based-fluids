cmake_minimum_required(VERSION 3.2)

project(position-based-fluids)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/CompactNSearch/include)

#include source code
file(GLOB H_FILES ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)



# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
    finite-diff
    GIT_REPOSITORY https://github.com/zfergus/finite-diff.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(finite-diff)

# Polyscope 
add_subdirectory("deps/polyscope")

#add_subdirectory("deps/CompactNSearch")

#build executable 
add_executable(${PROJECT_NAME} main.cpp ${CPP_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC polyscope Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PUBLIC finitediff::finitediff)

add_executable(main_2d main_2d.cpp ${CPP_FILES})
target_link_libraries(main_2d PUBLIC polyscope Eigen3::Eigen)# CompactNSearch)
target_link_libraries(main_2d PUBLIC finitediff::finitediff)
target_link_libraries(main_2d PUBLIC polyscope Eigen3::Eigen)

add_executable(smoke_sim smoke_sim.cpp ${CPP_FILES})
target_link_libraries(smoke_sim PUBLIC polyscope Eigen3::Eigen)# CompactNSearch)
target_link_libraries(smoke_sim PUBLIC finitediff::finitediff)
target_link_libraries(smoke_sim PUBLIC polyscope Eigen3::Eigen)
