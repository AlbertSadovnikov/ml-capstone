cmake_minimum_required(VERSION 3.7)
project(traffic)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0" CACHE STRING "Path to cuda libraries")
set(OpenCV_DIR "/opt/opencv/share/OpenCV" CACHE STRING "Path to OpenCV libraries")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(BOOST_ROOT "/opt/boost" CACHE STRING "Path to boost libraries")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Threads REQUIRED)
find_package(Boost 1.60 COMPONENTS program_options system filesystem REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(traffic ${SOURCE_FILES})
target_link_libraries(traffic ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
