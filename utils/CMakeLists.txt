cmake_minimum_required(VERSION 3.5.1)

# set the project name
project(gstreamer_pipeline)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")


# add the executable
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES gstreamer_pipeline.c)

add_executable(gstreamer_pipeline ${SOURCES})
target_include_directories(gstreamer_pipeline 
    PUBLIC "/usr/include/gstreamer-1.0"
    PUBLIC "/usr/include/glib-2.0/"
    PUBLIC "/usr/lib/x86_64-linux-gnu/glib-2.0/include")
target_link_libraries(gstreamer_pipeline 
    PUBLIC "gstreamer-1.0"
    PUBLIC "glib-2.0"
    PUBLIC "gobject-2.0")
