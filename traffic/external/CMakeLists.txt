option(BUILD_STATIC_LIB "Build the static library" ON)
option(BUILD_SHARED_LIB "Build the shared library" OFF)
option(BUILD_EXECUTABLE "Build darknet executable" OFF)
option(WITH_CUDA "with cuda" ON)
option(WITH_CUDNN "with cudnn" ON)
option(WITH_OPENCV "with opencv" ON)
option(WITH_OPENMP "with openmp" ON)

add_subdirectory(darknet)
set(DARKNET_INCLUDE_DIRS ${DARKNET_INCLUDE_DIRS}/darknet/include PARENT_SCOPE)
set(DARKNET_LIBRARY ${DARKNET_LIBRARY_STATIC} PARENT_SCOPE)

