#!/bin/sh
set -x
c++ -std=c++14 -o clrfft cl_fft.cpp test_rfft.cpp -I. -framework OpenCL -Wno-deprecated-declarations
c++ -std=c++14 -o clcfft cl_fft.cpp test_cfft.cpp -I. -framework OpenCL -Wno-deprecated-declarations

