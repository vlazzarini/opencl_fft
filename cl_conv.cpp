////////////////////////////////////////////////////////////////////////////////
// OpenCL Partitioned Convolution implementation
// Copyright (C) 2019 V Lazzarini
//
// This software is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
//
////////////////////////////////////////////////////////////////////////////////
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
kernel void reorder(global cmplx *out, global cmplx *in, global const int *b, int offs) {
   int k = get_global_id(0);
   out += offs;
   out[k] = in[b[k]]; 
   in[b[k]] = 0.f;
}
/* fft stage  */
kernel void fft(global cmplx *s, global const cmplx *w, int N, int n2, int offs) {
 int k, i, m, n;
 cmplx e, o;
 s += offs;
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
kernel void r2c(global cmplx *c, global const cmplx *w, int N, int offs) {
  int i = get_global_id(0);
  if(!i) {
   c[0] = (cmplx) ((c[0].x + c[0].y)*.5f, (c[0].x - c[0].y)*.5f);
   return;
  }
  int j = N - i;
  cmplx e, o, cj = conjg(c[j]), p;
  c += offs;
  e = .5f*(c[i] + cj);
  o = .5f*rot(cj - c[i]);
  p = prod(w[i], o); 
  c[i] = e + p;
  c[j] = conjg(e - p);
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
inline int reorder(cl_mem *out, cl_mem *in, cl_mem *b, int offs,
                   cl_command_queue commands, cl_kernel kern, size_t threads) {
  clSetKernelArg(kern, 3, sizeof(cl_int), &offs);
  clSetKernelArg(kern, 2, sizeof(cl_mem), b);
  clSetKernelArg(kern, 1, sizeof(cl_mem), in);
  clSetKernelArg(kern, 0, sizeof(cl_mem), out);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

inline int fft(cl_mem *data, cl_mem *w, int bins, int offs,
               cl_command_queue commands, cl_kernel kern, size_t threads) {
  int cl_err, n2;
  for (int n = 1; n < bins; n *= 2) {
    n2 = n << 1;
    clSetKernelArg(kern, 4, sizeof(cl_int), &offs);
    clSetKernelArg(kern, 3, sizeof(cl_int), &n2);
    clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
    clSetKernelArg(kern, 1, sizeof(cl_mem), w);
    clSetKernelArg(kern, 0, sizeof(cl_mem), data);
    cl_err = clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                    NULL, NULL);
  }
  return cl_err;
}

inline int real_cmplx(cl_mem *data, cl_mem *w, int bins, int offs,
                      cl_command_queue commands, cl_kernel kern,
                      size_t threads) {
  clSetKernelArg(kern, 3, sizeof(cl_int), &offs);
  clSetKernelArg(kern, 2, sizeof(cl_int), &bins);
  clSetKernelArg(kern, 1, sizeof(cl_mem), w);
  clSetKernelArg(kern, 0, sizeof(cl_mem), data);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

inline int cmplx_real(cl_mem *data, cl_mem *w, int bins,
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

inline int ola(cl_mem *out, cl_mem *in, int parts, cl_command_queue commands,
               cl_kernel kern, size_t threads) {
  clSetKernelArg(kern, 2, sizeof(cl_int), &parts);
  clSetKernelArg(kern, 1, sizeof(cl_mem), in);
  clSetKernelArg(kern, 0, sizeof(cl_mem), out);
  return clEnqueueNDRangeKernel(commands, kern, 1, NULL, &threads, NULL, 0,
                                NULL, NULL);
}

Clpconv::Clpconv(cl_device_id device_id, int cvs, int pts,
                 void (*errs)(std::string s, void *d), void *uData, void *inp1,
                 void *inp2, void *outp)
    : N(pts << 1), bins(pts), bsize((cvs / pts) * bins), nparts(cvs / pts),
      wp(0), wp2(nparts - 1), w{NULL, NULL}, w2{NULL, NULL}, b(NULL), in1(NULL),
      in2(NULL), out(NULL), spec1(NULL), spec2(NULL), olap(NULL), context(NULL),
      commands1(NULL), commands2(NULL), program(NULL), fft_kernel1(NULL),
      reorder_kernel1(NULL), fft_kernel2(NULL), reorder_kernel2(NULL),
      r2c_kernel1(NULL), r2c_kernel2(NULL), c2r_kernel(NULL),
      convol_kernel(NULL), olap_kernel(NULL),
      err(errs == NULL ? this->msg : errs), userData(uData), cl_err(CL_SUCCESS),
      host_mem(((uintptr_t)inp1 & (uintptr_t)inp2 & (uintptr_t)out) ? 1 : 0) {

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_err);
  if (!context) {
    err(cl_error_string(cl_err), userData);
    return;
  }

  /* two command queues for task parallelism */
  commands1 = clCreateCommandQueue(context, device_id, 0, &cl_err);
  if (!commands1) {
    err(cl_error_string(cl_err), userData);
    return;
  }
  commands2 = clCreateCommandQueue(context, device_id, 0, &cl_err);
  if (!commands2) {
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

  /* separate kernel objects for each command queue*/
  reorder_kernel1 = clCreateKernel(program, "reorder", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  fft_kernel1 = clCreateKernel(program, "fft", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  r2c_kernel1 = clCreateKernel(program, "r2c", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  reorder_kernel2 = clCreateKernel(program, "reorder", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  fft_kernel2 = clCreateKernel(program, "fft", &cl_err);
  if (cl_err != 0)
    err(cl_error_string(cl_err), userData);
  r2c_kernel2 = clCreateKernel(program, "r2c", &cl_err);
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

  in1 = clCreateBuffer(context, in1 ? CL_MEM_USE_HOST_PTR : 0,
                       bins * sizeof(cl_float2), inp1, &cl_err);
  in2 = clCreateBuffer(context, in2 ? CL_MEM_USE_HOST_PTR : 0,
                       bins * sizeof(cl_float2), inp2, &cl_err);
  olap = clCreateBuffer(context, out ? CL_MEM_USE_HOST_PTR : 0,
                        bins * sizeof(cl_float2), outp, &cl_err);
  out = clCreateBuffer(context, 0, bins * sizeof(cl_float2), NULL, &cl_err);
  spec1 = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                         NULL, &cl_err);
  spec2 = clCreateBuffer(context, CL_MEM_READ_ONLY, bsize * sizeof(cl_float2),
                         NULL, &cl_err);
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
  /* twiddle */
  std::vector<std::complex<float>> wd(bins);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * 2 * PI / bins));
    wd[i].imag(-sin(i * 2 * PI / bins));
  }
  clEnqueueWriteBuffer(commands1, w[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * 2 * PI / bins));
    wd[i].imag(sin(i * 2 * PI / bins));
  }
  clEnqueueWriteBuffer(commands1, w[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);

  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * PI / bins));
    wd[i].imag(-sin(i * PI / bins));
  }
  clEnqueueWriteBuffer(commands1, w2[0], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);
  for (int i = 0; i < bins; i++) {
    wd[i].real(cos(i * PI / bins));
    wd[i].imag(sin(i * PI / bins));
  }
  clEnqueueWriteBuffer(commands1, w2[1], CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)wd.data(), 0, NULL, NULL);

  /* bit-reversed indices */
  std::vector<int> bp(bins);
  for (int i = 0; i < bins; i++)
    bp[i] = i;
  for (int i = 1, n = bins / 2; i<bins; i = i << 1, n = n>> 1)
    for (int j = 0; j < i; j++)
      bp[i + j] = bp[j] + n;
  clEnqueueWriteBuffer(commands1, b, CL_TRUE, 0, sizeof(cl_int) * bins,
                       (const void *)bp.data(), 0, NULL, NULL);

  std::vector<float> zeros(bsize * 2, 0);
  clEnqueueWriteBuffer(commands1, olap, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands1, spec1, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands1, spec2, CL_TRUE, 0, sizeof(cl_float2) * bsize,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands1, in1, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
  clEnqueueWriteBuffer(commands1, in2, CL_TRUE, 0, sizeof(cl_float2) * bins,
                       (const void *)zeros.data(), 0, NULL, NULL);
}

Clpconv::~Clpconv() {
  clReleaseMemObject(w2[0]);
  clReleaseMemObject(w2[1]);
  clReleaseMemObject(w[0]);
  clReleaseMemObject(w[1]);
  clReleaseMemObject(b);
  clReleaseMemObject(out);
  clReleaseMemObject(in2);
  clReleaseMemObject(in1);
  clReleaseMemObject(olap);
  clReleaseMemObject(spec1);
  clReleaseMemObject(spec2);
  clReleaseKernel(fft_kernel2);
  clReleaseKernel(reorder_kernel2);
  clReleaseKernel(r2c_kernel2);
  clReleaseKernel(fft_kernel1);
  clReleaseKernel(reorder_kernel1);
  clReleaseKernel(r2c_kernel1);
  clReleaseKernel(c2r_kernel);
  clReleaseKernel(olap_kernel);
  clReleaseKernel(convol_kernel);
  clReleaseCommandQueue(commands1);
  clReleaseCommandQueue(commands2);
  clReleaseContext(context);
}

int Clpconv::push_ir(float *ir) {
  size_t bytes = sizeof(cl_float2) * bins;
  size_t threads;
  int n2;
  for (int i = 0; i < nparts; i++) {
    if (!host_mem)
      cl_err = clEnqueueWriteBuffer(commands1, in2, CL_TRUE, 0, bytes >> 1,
                                    &ir[i * bins], 0, NULL, NULL);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err =
        reorder(&spec2, &in2, &b, wp2 * bins, commands1, reorder_kernel1, bins);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err =
        fft(&spec2, &w[0], bins, wp2 * bins, commands1, fft_kernel1, bins >> 1);
    if (cl_err != CL_SUCCESS)
      return cl_err;
    cl_err = real_cmplx(&spec2, &w2[0], bins, wp2 * bins, commands1,
                        r2c_kernel1, bins >> 1);
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
    cl_err = clEnqueueWriteBuffer(commands1, in1, 0, 0, bytes >> 1, input, 0,
                                  NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      reorder(&spec1, &in1, &b, wp * bins, commands1, reorder_kernel1, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      fft(&spec1, &w[0], bins, wp * bins, commands1, fft_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&spec1, &w2[0], bins, wp * bins, commands1, r2c_kernel1,
                      bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  wp = wp != nparts - 1 ? wp + 1 : 0;
  cl_err = convol(&in1, &spec1, &spec2, wp, bins, nparts, commands1,
                  convol_kernel, bsize);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = cmplx_real(&in1, &w2[1], bins, commands1, c2r_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&out, &in1, &b, 0, commands1, reorder_kernel1, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&out, &w[1], bins, 0, commands1, fft_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = ola(&olap, &out, bins, commands1, olap_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueReadBuffer(commands1, olap, CL_TRUE, 0, bytes >> 1,
                                 output, 0, NULL, NULL);
  return cl_err;
}

int Clpconv::convolution(float *output, float *input1, float *input2) {
  size_t bytes = sizeof(cl_float2) * bins;
  if (!host_mem)
    cl_err = clEnqueueWriteBuffer(commands1, in1, 0, 0, bytes >> 1, input1, 0,
                                  NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueWriteBuffer(commands2, in2, 0, 0, bytes >> 1, input2, 0,
                                  NULL, NULL);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      reorder(&spec1, &in1, &b, wp * bins, commands1, reorder_kernel1, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      reorder(&spec2, &in2, &b, wp2 * bins, commands2, reorder_kernel2, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      fft(&spec1, &w[0], bins, wp * bins, commands1, fft_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err =
      fft(&spec2, &w[0], bins, wp2 * bins, commands2, fft_kernel2, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&spec1, &w2[0], bins, wp * bins, commands1, r2c_kernel1,
                      bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = real_cmplx(&spec2, &w2[0], bins, wp2 * bins, commands2, r2c_kernel2,
                      bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  wp = wp != nparts - 1 ? wp + 1 : 0;
  wp2 = wp2 == 0 ? nparts - 1 : wp2 - 1;
  clFinish(commands2);
  cl_err = convol(&in1, &spec1, &spec2, wp, bins, nparts, commands1,
                  convol_kernel, bsize);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = cmplx_real(&in1, &w2[1], bins, commands1, c2r_kernel, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = reorder(&out, &in1, &b, 0, commands1, reorder_kernel1, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = fft(&out, &w[1], bins, 0, commands1, fft_kernel1, bins >> 1);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  cl_err = ola(&olap, &out, bins, commands1, olap_kernel, bins);
  if (cl_err != CL_SUCCESS)
    return cl_err;
  if (!host_mem)
    cl_err = clEnqueueReadBuffer(commands1, olap, CL_TRUE, 0, bytes >> 1,
                                 output, 0, NULL, NULL);
  return cl_err;
}
}
