cmake_minimum_required(VERSION 3.5.1)

find_package(PkgConfig)

# set the project name
project(gstreamer_pipeline)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

pkg_check_modules(GLIB2 glib-2.0)

# add the executable
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES gstreamer_pipeline.c)

add_executable(gstreamer_pipeline ${SOURCES})
target_include_directories(gstreamer_pipeline 
    PUBLIC "/usr/include/gstreamer-1.0"
    ${GLIB2_INCLUDE_DIRS})
target_link_libraries(gstreamer_pipeline 
    PUBLIC "gstreamer-1.0"
    PUBLIC "gobject-2.0"
    ${GLIB2_LIBRARIES})
