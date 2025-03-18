find_package(CSOUND)
if(NOT CSOUND_INCLUDE_DIRS)
    message(FATAL_ERROR "Csound header files are required")
endif()

add_library(libocl_conv SHARED opcode.cpp)
target_link_libraries(libocl_conv  libcl_conv)
target_include_directories(libocl_conv PRIVATE ${CSOUND_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
set_property(TARGET libocl_conv PROPERTY CXX_STANDARD 11)
