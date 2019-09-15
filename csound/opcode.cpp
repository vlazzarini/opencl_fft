/*
  opcode.cpp: open cl convolution opcodes

  Copyright (C) 2019 Victor Lazzarini
  This file is part of Csound.

  The Csound Library is free software; you can redistribute it
  and/or modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  Csound is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with Csound; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA
*/

#include <cl_conv.h>
#include <cl_dconv.h>
#include <modload.h>
#include <plugin.h>
#include <vector>

namespace csnd {
void err_msg(std::string s, void *uData) {
  Csound *cs = (Csound *)uData;
  cs->message(s);
}

struct Conv : Plugin<1, 6> {
  cl_conv::Clpconv *clpconv;
  cl_conv::Cldconv *cldconv;
  Table ir;
  int parts, cnt;
  bool dconv;
  csnd::AuxMem<float> bufin, bufout;

  int init() {
    int size;
    int err;
    cl_device_id device_ids[32], id;
    cl_uint num = 0, nump = 0;
    char name[128];

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 32, device_ids, &num);
    if (err != CL_SUCCESS)
      return csound->init_error("failed to find an OpenCL device!\n");
    id = device_ids[(int)inargs[3]];
    clGetDeviceInfo(id, CL_DEVICE_NAME, 128, name, NULL);
    csound->message("using device: ");
    csound->message(name);
    ir.init(csound, inargs(1));
    parts = inargs[2];
    size = inargs[5] == 0 ? ir.len() : inargs[5];
    size -= inargs[4];
    MYFLT _0dbfs = csound->_0dbfs();
    dconv = parts == 1 ? true : false;
    if (dconv) {
      int ksmps = insdshead->ksmps;
      cldconv = new cl_conv::Cldconv(id, size, ksmps, err_msg, (void *)csound);
      if (cldconv->get_cl_err() == CL_SUCCESS) {
        std::vector<float> coefs(size);
        for (int i = 0; i < size; i++)
          coefs[i] = ir[i] * _0dbfs;
        if (cldconv->push_ir(coefs.data()) == CL_SUCCESS) {
          bufout.allocate(csound, ksmps);
          bufin.allocate(csound, ksmps);
          cnt = 0;
          csound->plugin_deinit(this);
          return OK;
        }
        csound->message("error setting impulse response");
      }
      delete cldconv;
    } else {
      clpconv = new cl_conv::Clpconv(id, size, parts, err_msg, (void *)csound);
      if (clpconv->get_cl_err() == CL_SUCCESS) {
        std::vector<float> coefs(size);
        for (int i = 0; i < size; i++)
          coefs[i] = ir[i] * _0dbfs;
        if (clpconv->push_ir(coefs.data()) == CL_SUCCESS) {
          bufout.allocate(csound, parts);
          bufin.allocate(csound, parts);
          cnt = 0;
          csound->plugin_deinit(this);
          return OK;
        }
        csound->message("error setting impulse response");
      }
      delete clpconv;
    }
    return csound->init_error("error initialising opencl object");
  }

  int deinit() {
    if (dconv)
      delete cldconv;
    else
      delete clpconv;
    return OK;
  }

  int aperf() {
    AudioSig asig(this, inargs(0));
    AudioSig aout(this, outargs(0));

    if (dconv) {
      for (int n = offset; n < nsmps; n++)
        bufin[n] = (float)asig[n];
      if (cldconv->convolution(bufout.data(), bufin.data()) != CL_SUCCESS)
        return csound->perf_error("error computing convolution\n", this);
      for (int n = offset; n < nsmps; n++)
        aout[n] = (MYFLT)bufout[n];
    } else {
      for (int n = offset; n < nsmps; n++) {
        bufin[cnt] = (float)asig[n];
        aout[n] = (MYFLT)bufout[cnt];
        if (++cnt == parts) {
          if (clpconv->convolution(bufout.data(), bufin.data()) != CL_SUCCESS)
            return csound->perf_error("error computing convolution\n", this);
          cnt = 0;
        }
      }
    }
    return OK;
  }
};

struct TVConv : Plugin<1, 7> {
  cl_conv::Clpconv *clpconv;
  cl_conv::Cldconv *cldconv;
  int parts, cnt;
  bool dconv;
  csnd::AuxMem<float> bufin1, bufin2, bufout;

  int init() {
    int size;
    int err;
    cl_device_id device_ids[32], id;
    cl_uint num = 0, nump = 0;
    char name[128];

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 32, device_ids, &num);
    if (err != CL_SUCCESS)
      return csound->init_error("failed to find an OpenCL device!\n");
    id = device_ids[(int)inargs[6]];
    clGetDeviceInfo(id, CL_DEVICE_NAME, 128, name, NULL);
    csound->message("using device: ");
    csound->message(name);
    size = inargs[5];
    parts = inargs[4];
    dconv = parts == 1 ? true : false;
    if (dconv) {
      int ksmps = insdshead->ksmps;
      cldconv = new cl_conv::Cldconv(id, size, ksmps, err_msg, (void *)csound);
      if (cldconv->get_cl_err() == CL_SUCCESS) {
        bufout.allocate(csound, ksmps);
        bufin1.allocate(csound, ksmps);
        bufin2.allocate(csound, ksmps);
        cnt = 0;
        csound->plugin_deinit(this);
        return OK;
      }
      delete cldconv;
    } else {
      clpconv = new cl_conv::Clpconv(id, size, parts, err_msg, (void *)csound);
      if (clpconv->get_cl_err() == CL_SUCCESS) {
        cnt = 0;
        bufout.allocate(csound, parts);
        bufin1.allocate(csound, parts);
        bufin2.allocate(csound, parts);
        csound->plugin_deinit(this);
        return OK;
      }
      delete clpconv;
    }
    return csound->init_error("error initialising opencl object");
  }

  int deinit() {
    if (dconv)
      delete cldconv;
    else
      delete clpconv;
    return OK;
  }

  int aperf() {
    AudioSig asig1(this, inargs(0));
    AudioSig asig2(this, inargs(1));
    AudioSig aout(this, outargs(0));
    int frz1 = (int)inargs[2], frz2 = (int)inargs[2];
    MYFLT _0dbfs = csound->_0dbfs();

    if (dconv) {
      for (int n = offset; n < nsmps; n++) {
        bufin1[n] = (float)frz1 ? (float)(asig1[n] / _0dbfs) : bufin1[cnt];
        bufin2[n] = (float)frz1 ? (float)(asig2[n] / _0dbfs) : bufin2[cnt];
      }
      if (cldconv->convolution(bufout.data(), bufin1.data(), bufin2.data()) !=
          CL_SUCCESS)
        return csound->perf_error("error computing convolution\n", this);
      for (int n = offset; n < nsmps; n++)
        aout[n] = bufout[n] * _0dbfs;
    } else {
      for (int n = offset; n < nsmps; n++) {
        bufin1[cnt] = frz1 ? (float)(asig1[n] / _0dbfs) : bufin1[cnt];
        bufin2[cnt] = frz1 ? (float)(asig2[n] / _0dbfs) : bufin2[cnt];
        aout[n] = bufout[cnt] * _0dbfs;
        if (++cnt == parts) {
          if (clpconv->convolution(bufout.data(), bufin1.data(),
                                   bufin2.data()) != CL_SUCCESS)
            return csound->perf_error("error computing convolution\n", this);
          cnt = 0;
        }
      }
    }
    return OK;
  }
};

void on_load(Csound *csound) {
  plugin<Conv>(csound, "clconv", "a", "aiiioo", csnd::thread::ia);
  plugin<TVConv>(csound, "cltvconv", "a", "aakkiii", csnd::thread::ia);
}
}
