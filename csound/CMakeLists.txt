find_package(CSOUND)
if(NOT CSOUND_INCLUDE_DIRS)
    message(FATAL_ERROR "Csound header files are required")
endif()

add_library(libcl_conv_opc SHARED opcode.cpp)
target_link_libraries(libcl_conv_opc  libcl_conv)
target_include_directories(libcl_conv_opc PRIVATE ${CSOUND_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
set_property(TARGET libcl_conv_opc PROPERTY CXX_STANDARD 11)
