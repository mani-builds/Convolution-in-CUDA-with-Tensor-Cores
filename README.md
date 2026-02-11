A convolutional layer from scratch in CUDA, optimize for GPU architectures with tensor cores 

* Naive Implementation of a 2D/3D convolutional kernel in CUDA
It uses Global Memory
Profile: High Latency and Low Occupancy

* Basic Optimizations 
Constant memory for the Kernel (since the data doesn't change during execution)
Shared Memory Tiling. Reduces global accesses dramatically 11x fewer
Profile these.

* im2col - convert an image into an array by flattening it, so that a GEMM can be used instead of sliding filter


TODO
* [ ] Tensor core GEMM with CUDA WMMA API (Warp Matrix Multiply-Accumulate) for FP16 tensor ops

* [ ] Baseline comparison with cuDNN's convolution

* [ ] Make it an executable in linux like a library

* [ ] Make it importable with pytorch
