/////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution implementation
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
/////////////////////////////////////////////////////////////////////
#include "cl_conv.h"
#include <vector>

namespace cl_conv {

const double PI = 3.141592653589793;

const char *pconvcode = R"(
/* complex type */
typedef float2 cmplx;
/* complex product */
inline cmplx prod(cmplx a, cmplx b){
     return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); 
}
/* complex conj */
inline cmplx conjg(cmplx a) {
    return (cmplx) (a.x, - a.y);
}
/* rotation by pi */
inline cmplx rot(cmplx a) {
   return (cmplx) (-a.y, a.x);
} 
/* atomic add */
inline void AtomicAdd(volatile __global float *source, const float operand) {
     union {
         uint intVal;
         float floatVal;
     } newVal;
     union {
         uint intVal;
         float floatVal;
     } prevVal;
     do {
         prevVal.floatVal = *source;
         newVal.floatVal = prevVal.floatVal + operand;
     } while (atomic_cmpxchg((volatile __global uint *) source, 
       prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
/* data reordering */
inline void rod(global cmplx *out, global cmplx *in, global const int *b) {
   int k = get_global_id(0);
   out[k] = in[b[k]]; 
   in[b[k]] = 0.f;
}
/* fft stage  */
inline void dft(global cmplx *s, global const cmplx *w, int N, int n2) {
 int k, i, m, n;
 cmplx e, o;
 k = get_global_id(0)*n2;
 m = k/N; 
 n = n2 >> 1;
 k =  k%N + m;
 i = k + n;
 e = s[k];
 o = prod(s[i],w[m*N/n2]);
 s[k] = e + o;
 s[i] = e - o;  
}
/* rfft conversion */
inline void rcp(global cmplx *c, global const cmplx *w, int N) {
  int i = get_global_id(0);
  if(!i) {
   c[0] = (cmplx) ((c[0].x + c[0].y)*.5f, (c[0].x - c[0].y)*.5f);
   return;
  }
  int j = N - i;
  cmplx e, o, cj = conjg(c[j]), p;
  e = .5f*(c[i] + cj);
  o = .5f*rot(cj - c[i]);
  p = prod(w[i], o); 
  c[i] = e + p;
  c[j] = conjg(e - p);
}
/* TWO of each reorder, fft and r2c kernels for
   task parallelism
*/
kernel void reorder(global cmplx *out, global cmplx *in, global const int *b) {
   rod(out, in, b);
}
kernel void reorder1(global cmplx *out, global cmplx *in, global const int *b) {
   rod(out, in, b);
}
kernel void fft(global cmplx *s, global const cmplx *w, int N, int n2) {
  dft(s, w, N, n2); 
}
kernel void fft1(global cmplx *s, global const cmplx *w, int N, int n2) {
  dft(s, w, N, n2); 
}
kernel void r2c(global cmplx *c, global const cmplx *w, int N) {
  rcp(c, w, N);
}
kernel void r2c1(global cmplx *c, global const cmplx *w, int N) {
  rcp(c, w, N);
}
/* inverse rfft conversion */
kernel void c2r(global cmplx *c, global const cmplx *w, int N) {
  int i = get_global_id(0);
  if(!i) {
   c[0] = (cmplx) ((c[0].x + c[0].y), (c[0].x - c[0].y));
   return; 
  }
  int j = N - i;
  cmplx e, o, cj = conjg(c[j]), p;
  e = .5f*(c[i] + cj);
  o = .5f*rot(c[i] - cj);
  p = prod(w[i], o);
  c[i] = e + p;
  c[j] = conjg(e - p); 
}
/* convolution */
kernel void convol(global float *out, global const cmplx *in, 
		    global const cmplx *coef, int rp, int b, 
		    int nparts) {
  /* thread count */
  int k = get_global_id(0); /* bin pos */ 
  int n = k%b;  /* inframe pos   */
  int n2 = n << 1;
  cmplx s;
  rp += k/b;       /*  rp pos */
  /* select correct input buffer */
  in += (rp < nparts ? rp : rp - nparts)*b;  
  /* complex multiplication + sums */
  s = n ? prod(in[n], coef[k]) : 
             (cmplx) (in[0].x*coef[k].x, in[0].y*coef[k].y);
  AtomicAdd(&out[n2], s.x);  
  AtomicAdd(&out[n2+1], s.y);                                
}  
/* sample-by-sample overlap-add operation */
kernel void olap(global float *buf, global const float *in, int parts){
   int n = get_global_id(0);
   buf[n] = (in[n] + buf[parts+n])/parts;
   buf[parts+n] = in[parts+n];
}
)";

/*  kernel dispatch functions */
inline int reorder(cl_mem *out, cl_mem *in, cl_mem *b,
                   cl_command_queue commands, cl_kernel kern, size_t threads) {
  clSetKernelArg(kern, 2, sizeof(cl_mem), b);
  clSetKernelArg(kern, 1, sizeof(cl_mem), in);
  clSetKernelArg(kern, 0, sizeof(cl_mem), out);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

inline int fft(cl_mem *data, cl_mem *w, int bins, cl_command_queue commands,
               cl_kernel kern, size_t threads) {
  int cl_err, n2;
  for (int n = 1; n < bins; n *= 2) {
    n2 = n << 1;
    clSetKernelArg(kern, 3, sizeof(cl_int), &n2);
    clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
    clSetKernelArg(kern, 1, sizeof(cl_mem), w);
    clSetKernelArg(kern, 0, sizeof(cl_mem), data);
    cl_err = clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                    NULL, NULL);
  }
  return cl_err;
}

inline int real_cmplx(cl_mem *data, cl_mem *w, int bins,
                      cl_command_queue commands, cl_kernel kern,
                      size_t threads) {
  clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
  clSetKernelArg(kern, 1, sizeof(cl_mem), w);
  clSetKernelArg(kern, 0, sizeof(cl_mem), data);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

inline int convol(cl_mem *out, cl_mem *in, cl_mem *coefs, int wp, int bins,
                  int nparts, cl_command_queue commands, cl_kernel kern,
                  size_t threads) {
  clSetKernelArg(kern, 5, sizeof(cl_int), &nparts);
  clSetKernelArg(kern, 4, sizeof(cl_int), &bins);
  clSetKernelArg(kern, 3, sizeof(cl_int), &wp);
  clSetKernelArg(kern, 2, sizeof(cl_mem), coefs);
  clSetKernelArg(kern, 1, sizeof(cl_mem), in);
  clSetKernelArg(kern, 0, sizeof(cl_mem), out);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

inline int olap(cl_mem *out, cl_mem *in, int parts, cl_command_queue commands,
                cl_kernel kern, size_t threads) {
  clSetKernelArg(kern, 2, sizeof(cl_int), &parts);
  clSetKernelArg(kern, 1, sizeof(cl_mem), in);
  clSetKernelArg(kern, 0, sizeof(cl_mem), out);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

Clpconv::Clpconv(cl_device_id device_id, int cvs, int pts,
                 void (*errs)(std::string s, void *d), void *uData, void *in1,
                 void *in2, void *out)
    : N(pts << 1), bins(pts), bsize((cvs / pts) * bins), nparts(cvs / pts),
      wp(0), wp2(nparts - 1), w{NULL, NULL}, w2{NULL, NULL}, b(NULL),
      specin(NULL), specout(NULL), specin1(NULL), specout1(NULL), buff(NULL),
      coefs(NULL), in(NULL), context(NULL), commands(NULL), commands1(NULL),
      program(NULL), fft_kernel(NULL), reorder_kernel(NULL), fft_kernel1(NULL),
      reorder_kernel1(NULL), r2c_kernel(NULL), r2c_kernel1(NULL),
      c2r_kernel(NULL), convol_kernel(NULL), olap_kernel(NULL),
      err(errs == NULL ? this->msg : errs), userData(uData), cl_err(CL_SUCCESS),
      host_mem(((uintptr_t)in1 & (uintptr_t)in2 & (uintptr_t)out) ? 1 : 0) {

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_err);
  if (!context) {
    err(cl_error_string(cl_err), userData);
    return;
  }
  commands = clCreateCommandQueue(context, device_id, 0, &cl_err);
  if (!commands) {
    err(cl_error_string(cl_err), userData);
    return;
  }

  commands1 = clCreateCommandQueue(context, device_id, 0, &cl_err);
  if (!commands1) {
    err(cl_error_string(cl_err), userData);
    return;
  }

  program = clCreateProgramWithSource(context, 1, (const char **)&pconvcode,
                                      NULL, &cl_err);
  if (!program) {
    err("error creating opencl program\n", userData);
    err(cl_error_string(cl_err), userData);
    return;
  }
  cl_err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (cl_err) {
    char log[2048];
    size_t llen;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, &llen);
    err("error building opencl program\n", userData);
    err(cl_error_string(cl_err), userData);
    err(log, userData);
    return;
  }
  reorder_kernel = clCreateKernel(program, "reorder", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  fft_kernel = clCreateKernel(program, "fft", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  r2c_kernel = clCreateKernel(program, "r2c", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  reorder_kernel1 = clCreateKernel(program, "reorder1", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  fft_kernel1 = clCreateKernel(program, "fft1", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  r2c_kernel1 = clCreateKernel(program, "r2c1", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);

  c2r_kernel = clCreateKernel(program, "c2r", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  convol_kernel = clCreateKernel(program, "convol", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  olap_kernel = clCreateKernel(program, "olap", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);

  specout = clCreateBuffer(context, 0, bins * sizeof(cl_float2), NULL, &cl_err);
  specout1 =
      clCreateBuffer(context, 0, bins * sizeof(cl_float2), NULL, &cl_err);
  specin = clCreateBuffer(context, in1 ? CL_MEM_USE_HOST_PTR : 0,
                          bins * sizeof(cl_float2), in1, &cl_err);
  specin1 = clCreateBuffer(context, in2 ? CL_MEM_USE_HOST_PTR : 0,
                           bins * sizeof(cl_float2), in2, &cl_err);
  w[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                        NULL, &cl_err);
  w[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                        NULL, &cl_err);
  w2[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                         NULL, &cl_err);
  w2[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_float2),
                         NULL, &cl_err);
  b = clCreateBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(cl_int), NULL,
                     NULL);
  coefs = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                         NULL, &cl_err);
  in = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                      NULL, &cl_err);
  buff = clCreateBuffer(context, out ? CL_MEM_USE_HOST_PTR : 0,
                        bins * sizeof(cl_float2), out, &cl_err);

  /* twiddle */
  std::vector<std::complex<float>> wd(bins);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * 2 * PI / bins));
    wd[i].imag(-sin(i * 2 * PI / bins));
  }
  clEnqueueWriteBuffer(commands, w[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * 2 * PI / bins));
    wd[i].imag(sin(i * 2 * PI / bins));
  }
  clEnqueueWriteBuffer(commands, w[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);

  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * PI / bins));
    wd[i].imag(-sin(i * PI / bins));
  }
  clEnqueueWriteBuffer(commands, w2[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * PI / bins));
    wd[i].imag(sin(i * PI / bins));
  }
  clEnqueueWriteBuffer(commands, w2[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);

  /* bit-reversed indices */
  std::vector<int> bp(bins);
  for (int i = 0; i < bins; i++)
    bp[i] = i;
  for (int i = 1, n = bins / 2; i<bins; i = i << 1, n = n>> 1)
    for (int j = 0; j < i; j++)
      bp[i + j] = bp[j] + n;
  clEnqueueWriteBuffer(commands, b, CL_TRUE, 0, sizeof(cl_int) * bins,
                       (const void *)bp.data(), 0, NULL, NULL);

  std::vector<float> zeros(bsize * 2, 0);
  clEnqueueWriteBuffer(commands, buff, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands, in, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands, coefs, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands, specin, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands, specin1, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
}

Clpconv::~Clpconv() {
  clReleaseMemObject(w2[0]);
  clReleaseMemObject(w2[1]);
  clReleaseMemObject(w[0]);
  clReleaseMemObject(w[1]);
  clReleaseMemObject(b);
  clReleaseMemObject(specout);
  clReleaseMemObject(specin);
  clReleaseMemObject(specout1);
  clReleaseMemObject(specin1);
  clReleaseMemObject(buff);
  clReleaseMemObject(coefs);
  clReleaseMemObject(in);
  clReleaseKernel(fft_kernel);
  clReleaseKernel(reorder_kernel);
  clReleaseKernel(r2c_kernel);
  clReleaseKernel(fft_kernel1);
  clReleaseKernel(reorder_kernel1);
  clReleaseKernel(r2c_kernel1);
  clReleaseKernel(c2r_kernel);
  clReleaseKernel(olap_kernel);
  clReleaseKernel(convol_kernel);
  clReleaseCommandQueue(commands);
  clReleaseCommandQueue(commands1);
  clReleaseContext(context);
}

int Clpconv::push_ir(float *ir) {
  size_t bytes = sizeof(cl_float2) * bins;
  size_t threads;
  int n2;
  for (int i = 0; i < nparts; i++) {
    if (!host_mem)
      cl_err = clEnqueueWriteBuffer(commands, specin, CL_TRUE, 0, bytes >> 1,
                                    &ir[i * bins], 0, NULL, NULL);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = reorder(&specout, &specin, &b, commands, reorder_kernel, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = fft(&specout, &w[0], bins, commands, fft_kernel, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err =
        real_cmplx(&specout, &w2[0], bins, commands, r2c_kernel, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = clEnqueueCopyBuffer(commands, specout, coefs, 0, bytes * wp2,
                                 bytes, 0, NULL, NULL);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;
  }
  return CL_SUCCESS;
}

int Clpconv::convolution(float *output, float *input) {
  int n2;
  size_t bytes = sizeof(cl_float2) * bins;
  char zro = 0;
  if (!host_mem)
    cl_err = clEnqueueWriteBuffer(commands, specin, CL_TRUE, 0, bytes >> 1,
                                  input, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&specout, &specin, &b, commands, reorder_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&specout, &w[0], bins, commands, fft_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&specout, &w2[0], bins, commands, r2c_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = clEnqueueCopyBuffer(commands, specout, in, 0, bytes * wp, bytes, 0,
                               NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  wp = wp != nparts - 1 ? wp + 1 : 0;
  cl_err = convol(&specin, &in, &coefs, wp, bins, nparts, commands,
                  convol_kernel, bsize);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&specin, &w2[1], bins, commands, c2r_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&specout, &specin, &b, commands, reorder_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&specout, &w[1], bins, commands, fft_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = olap(&buff, &specout, bins, commands, olap_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueReadBuffer(commands, buff, CL_TRUE, 0, bytes >> 1, output,
                                 0, NULL, NULL);
  return cl_err;
}

int Clpconv::convolution(float *output, float *input1, float *input2) {
  size_t bytes = sizeof(cl_float2) * bins;
  if (!host_mem)
    cl_err = clEnqueueWriteBuffer(commands, specin, CL_TRUE, 0, bytes >> 1,
                                  input1, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueWriteBuffer(commands1, specin1, CL_TRUE, 0, bytes >> 1,
                                  input2, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&specout, &specin, &b, commands, reorder_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&specout1, &specin1, &b, commands1, reorder_kernel1, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&specout, &w[0], bins, commands, fft_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&specout1, &w[0], bins, commands1, fft_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&specout, &w2[0], bins, commands, r2c_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      real_cmplx(&specout1, &w2[0], bins, commands1, r2c_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = clEnqueueCopyBuffer(commands, specout, in, 0, bytes * wp, bytes, 0,
                               NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  wp = wp != nparts - 1 ? wp + 1 : 0;
  cl_err = clEnqueueCopyBuffer(commands1, specout1, coefs, 0, bytes * wp2,
                               bytes, 0, NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;
  cl_err = convol(&specin, &in, &coefs, wp, bins, nparts, commands,
                  convol_kernel, bsize);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&specin, &w2[1], bins, commands, c2r_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&specout, &specin, &b, commands, reorder_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&specout, &w[1], bins, commands, fft_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = olap(&buff, &specout, bins, commands, olap_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueReadBuffer(commands, buff, CL_TRUE, 0, bytes >> 1, output,
                                 0, NULL, NULL);
  return cl_err;
}
}
