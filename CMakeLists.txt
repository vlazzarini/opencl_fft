cmake_minimum_required(VERSION 3.13)
project(OpenCL_FFT)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_VERBOSE_MAKEFILE 1)

set(SRCS cl_conv.cpp cl_dconv.cpp)
find_package(OpenCL REQUIRED )

add_library(libcl_conv SHARED ${SRCS})

if(APPLE)
target_link_libraries(libcl_conv "-framework OpenCL")
else()
target_link_libraries(libcl_conv OpenCL)
endif()

set_property(TARGET libcl_conv PROPERTY CXX_STANDARD 14)

add_subdirectory(csound)


