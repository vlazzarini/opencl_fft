/*
  clconv: open cl fast partitioned convolution

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

#include <modload.h>
#include <plugin.h>
#include <cl_conv.h>
#include <vector>

namespace csnd {
void err_msg(std::string s, void *uData) {
  Csound *cs = (Csound *)uData;
  cs->message(s);
}

struct PConv : Plugin<1, 6> {
  cl_conv::Clconv *clconv;
  Table ir;
  int parts, cnt;
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
    clconv = new cl_conv::Clconv(id, size, parts, err_msg, (void *)csound);
    if (clconv->get_cl_err() == CL_SUCCESS) {
      std::vector<float> coefs(size);
      for (int i = 0; i < size; i++)
        coefs[i] = ir[i];
      if (clconv->push_ir(coefs.data()) == CL_SUCCESS) {
        bufout.allocate(csound, parts);
        bufin.allocate(csound, parts << 1);
        cnt = 0;
        csound->plugin_deinit(this);
        return OK;
      }
      csound->message("error setting impulse response");
    }
    delete clconv;
    return csound->init_error("error initialising opencl object");
  }

  int deinit() {
    delete clconv;
    return OK;
  }

  int aperf() {
    AudioSig asig(this, inargs(0));
    AudioSig aout(this, outargs(0));

    for (int n = offset; n < nsmps; n++) {
      bufin[cnt] = (float)asig[n];
      aout[n] = (MYFLT)bufout[cnt];
      if (++cnt == parts) {
        if (clconv->convolution(bufout.data(), bufin.data()) != CL_SUCCESS)
          return csound->perf_error("error computing convolution\n", this);
        cnt = 0;
      }
    }
    return OK;
  }
};

void on_load(Csound *csound) {
  plugin<PConv>(csound, "clconv", "a", "aiiioo", csnd::thread::ia);
}
}
