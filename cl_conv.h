/////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution class
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#ifndef __CL_CONV_H__
#define __CL_CONV_H__
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#ifdef __MACH__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

namespace cl_conv {

const char *cl_string(int err);

class Clconv {
  int N, bins;
  int bsize, nparts, wp;
  cl_mem w[2], w2[2], b;
  cl_mem fftin, fftout;
  cl_mem buff, coefs, in;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel fft_kernel, reorder_kernel;
  cl_kernel r2c_kernel, c2r_kernel;
  cl_kernel convol_kernel, olap_kernel;
  size_t wgs, rwgs;
  size_t crwgs, rcwgs;
  size_t cvwgs, olwgs;
  void (*err)(std::string s, void *uData);
  void *userData;
  int cl_err;

  static void msg(std::string str, void *userData) {
    if (userData == NULL)
      std::cout << str << std::endl;
  }

  void fft(bool fwd);
  void rfft();
  void rifft();

public:
  /** Constructor \n
      device_id - OpenCL device ID \n
      cvs - impulse response size \n
      pts - partition size \n
      errs - error message callback \n
      uData - callback user data \n
  */
  Clconv(cl_device_id device_id, int cvs, int pts,
         void (*errs)(std::string s, void *d) = NULL, void *uData = NULL);
  ~Clconv();

  /** returns an error string relative to error code err */
  const char *cl_error_string(int err) { return cl_string(err); }

  /** set the convolution impulse response
      ir - impulse response of size cvs;
  */
  int push_ir(float *ir);

  /** Convolution computation
      output - output array (partition size samples) \n
      input - input array (partition size * 2, but
        holding only partition size samples, zero-padded to
        partition length)  \n
  */
  int convolution(float *output, float *input);

  /** Time-varying convolution computation
      output - output array (partition size samples) \n
      input1, input2 - input arrays (partition size * 2, but
        holding only partition size samples, zero-padded to
        partition length)  \n
  */
  int convolution(float *output, float *input1, float *input2);

  /** get a recorded error code, CL_SUCCESS if no error was recorded
   */
  int get_cl_err() { return cl_err; }
};

class Cldconv {
  int irsize, vsize, wp;
  cl_mem buff, coefs, del;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel convol_kernel;
  size_t wgs;
  void (*err)(std::string s, void *uData);
  void *userData;
  int cl_err;

  static void msg(std::string str, void *userData) {
    if (userData == NULL)
      std::cout << str << std::endl;
  }

public:
  /** Constructor \n
      device_id - OpenCL device ID \n
      cvs - impulse response size \n
      vsize - processing vector size \n
      errs - error message callback \n
      uData - callback user data \n
  */
  Cldconv(cl_device_id device_id, int cvs, int vsize,
         void (*errs)(std::string s, void *d) = NULL, void *uData = NULL);
  
  ~Cldconv();

  /** returns an error string relative to error code err */
  const char *cl_error_string(int err) { return cl_string(err); }

  /** set the convolution impulse response
      ir - impulse response of size cvs;
  */
  int push_ir(float *ir);

  /** Convolution computation
      output - output array (vsize samples) \n
      input - input array (vsize samples) \n
  */
  int convolution(float *output, float *input);

  int convolution(float *out, float *in1, float *in2);

  /** get a recorded error code, CL_SUCCESS if no error was recorded
   */
  int get_cl_err() { return cl_err; }
};
}
#endif
