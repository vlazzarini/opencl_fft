/////////////////////////////////////////////////////////////////////
// OpenCL 1-D Radix-2 FFT classes
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __CL_FFT_H__
#define __CL_FFT_H__

#include <complex>
#ifdef __MACH__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <iostream>

const double PI = 3.141592653589793;
const char * cl_error_string(int err);

/** Complex to Complex FFT class
 **/
class Clcfft {
  
 protected:
  int N;
  bool forward;
  cl_mem w, data;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel fft_kernel;
  size_t wgs;
  char log[2048];
  size_t  llen;

  int fft(std::complex<float> *c) {
    int err;
    for (int n = 1; n < N; n *= 2) {
      int n2 = n << 1;
      size_t threads = N >> 1;
      clSetKernelArg(fft_kernel, 3, sizeof(cl_int), &n2);
      err = clEnqueueNDRangeKernel(commands, fft_kernel,1, NULL, &threads,
                                   &wgs, 0, NULL, NULL);
      if(err)
        std::cout << "failed to run fft kernel" <<
          cl_error_string(err) << std::endl;
      clFinish(commands);
    }
    return err;
  }

  void reorder(std::complex<float> *s) {
    uint32_t j = 0, m;
    for (uint32_t i = 0; i < N; i++) {
      if (j > i) {
        std::swap(s[i], s[j]);
      }
      m = N / 2;
      while (m >= 2 && j >= m) {
        j -= m;
        m /= 2;
      }
      j += m;
    }
  }

 public:

  /** Constructor \n
      device_id - OpenCL device ID \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
   */
  Clcfft(cl_device_id device_id, int size, bool fwd=true);
  
  /** Destructor
   */
  virtual ~Clcfft();

  /** DFT operation (in-place) \n
      c - data array with N complex numbers \n
  */   
  int transform(std::complex<float> *c);   
};

/** Real to Complex FFT class
 **/
class Clrfft : public Clcfft {

  cl_mem w2;
  cl_kernel conv_kernel, iconv_kernel;
  size_t cwgs, iwgs;

  public:

  /** Constructor \n
      device_id - OpenCL device ID \n
      size - DFT size (N) \n
      fwd - direction (true: forward; false: inverse) \n
   */
  Clrfft(cl_device_id device_id, int size, bool dir);

  /** Destructor
   */
  virtual ~Clrfft();
  
  /** DFT operation \n
      c - data array with N/2 complex numbers \n
      r - data array with N real numbers \n
      Transform is in place if both c and r point to the same memory.\n
      If separate locations are used, r holds input data in forward transform \n
      and c will contain the output. For inverse, c is input, r is output.
  */   
  int transform(std::complex<float> *c, float *r); 

};

#endif
