/////////////////////////////////////////////////////////////////////
// OpenCL 1-D Radix-2 FFT test program - real to complex
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "cl_fft.h"
#include <iostream>
#include <iomanip>
#include <vector>

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

  Clrfft dft(device_ids[1],N,true);
  Clrfft idft(device_ids[1],N,false);
  
  std::vector<std::complex<float>> spec(N/2);
  std::vector<float> sig(N);
   
  for(int i = 0; i < N; i++){
    sig[i] = 0.5+(sin(i*2*PI/N)) + 0.5*cos(i*PI);
  }

  std::cout << "in =[";
  for(int i = 0; i < N-1; i++) 
    std::cout << sig[i] << ", ";
  std::cout <<  sig[N-1] << "]" << std::endl;
   
  dft.transform(spec.data(), sig.data());

 std::cout << "spec =[";
  for(int i = 0; i < N/2 - 1; i++) 
    std::cout << spec[i] << ","; 
  std::cout <<  spec[N/2 - 1] << "]" << std::endl;
  
 idft.transform(spec.data(), sig.data());

  std::cout << "out =[";
  for(int i = 0; i < N-1; i++) 
    std::cout << sig[i] << ",";
  std::cout <<  sig[N-1] << "]" << std::endl;
   
  return 0;
}
