#include "cl_fft.h"
/////////////////////////////////////////////////////////////////////
// OpenCL 1-D Radix-2 FFT test program - complex to complex
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <vector>

// test program
#define DEVID  1
#define N  16

int main() {
  int err;
  cl_device_id device_ids[32], device_id;
  cl_uint num = 0, nump =  0;
  cl_platform_id platforms[16];
  char name[128];
  std::cout << std::fixed;
  std::cout << std::setprecision(3);
    
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 32, device_ids, &num);
  if (err != CL_SUCCESS) {
        std::cout << "failed to find an OpenCL device!" <<
          cl_error_string(err) << std::endl;
        return -1;
  }
  clGetDeviceInfo(device_ids[DEVID], CL_DEVICE_NAME, 128, name, NULL);
  std::cout <<  "using device " << DEVID << ":" << name << std::endl;

  Clcfft dft(device_ids[1],N,true);
  Clcfft idft(device_ids[1],N,false);
  
  std::vector<std::complex<float>> sig(N);

   
  for(int i = 0; i < 16; i++){
    sig[i].real(sin(i*2*PI/16));
  }

  std::cout << "in =[";
  for(int i = 0; i < N-1; i++) 
    std::cout << sig[i].real() << ", ";
  std::cout <<  sig[N-1].real() << "]" << std::endl;
   
  dft.transform(sig.data());

 std::cout << "spec =[";
  for(int i = 0; i < N - 1; i++) 
    std::cout << sig[i] << ","; 
  std::cout <<  sig[N-1] << "]" << std::endl;
  
 idft.transform(sig.data());

  std::cout << "out =[";
  for(int i = 0; i < N-1; i++) 
    std::cout << sig[i].real() << ",";
  std::cout <<  sig[N-1].real() << "]" << std::endl;
   
  return 0;
}
