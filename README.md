OpenCL 1D Radix-2 FFT
================

This repository contains an implementation of the radix-2 FFT for
one-dimensional data. It contains code for both complex-to-complex
and real-to-complex operations. The main motivation is to provide support
for the parallelisatio of audio processing operations in OpenCL devices such
as GPUs, CPUs and accellerators.

The repository includes a complete implementation of fast partitioned
convolution, based on the FFT code. This is demonstrated in a csound
opcode with accompanying CSD code.

Victor Lazzarini, 2019.
