*** Project: GPU-Optimized Convolution from Scratch

#+BEGIN_QUOTE
This project implements 2D and 3D convolutional layers from scratch in CUDA, optimized for modern GPU architectures with tensor cores. It includes naive and optimized implementations, im2col transformation, and GEMM using Tensor cores (WMMA API), with profiling and comparison against cuDNN.
#+END_QUOTE

*** Features

+ *Naive Implementation*: 2D/3D convolution kernel in CUDA using global memory.  
  + Profile: /High latency, low occupancy/
+ *Basic Optimizations*:
  + Constant memory for kernel (since kernel data is static during execution)
  + Shared memory tiling: reduces global memory accesses by ~11x
  + Profiling with Nsight for bottlenecks
+ *im2col Transformation*:
  + Converts input images to column format for efficient GEMM
  + Implemented in RAW CUDA for GPU, and NumPy for CPU
+ *Tensor Core GEMM*:
  + Uses CUDA WMMA API for FP16 tensor operations
  + Leverages warp-level matrix multiply-accumulate for high throughput
+ *Comparison with cuDNN*:
  + Baseline performance and correctness checks
+ *Executable & Library*:
  + Can be built as a Linux executable and library
+ *PyTorch Integration*:
  + Planned for importable module

*** Implementation Details

+ *im2col*:
  + Converts input image to a padded matrix for GEMM
  + GPU kernel implemented in ~im2col.cu~
  + Example: For a 5x5 input and 3x3 kernel, im2col produces a 9x25 matrix
+ *GEMM with Tensor Cores*:
  + Implemented in ~gemm_tensors.cu~
  + Uses CUDA WMMA API for warp-level matrix multiplication
  + Supports large matrices (multiples of 16)
+ *Profiling*:
  + Nsight used to monitor occupancy, latency, and memory access patterns
+ *Testing*:
  + Synthetic data and MNIST
  + Verification against NumPy's ~convolve~

*** Example: im2col Kernel (CUDA)
#+BEGIN_SRC cuda
__global__ void im2col_kernel(float *image,int C_in, int H_in, int W_in, int im2col_H, int im2col_W, int kernel_size, float *im2col){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_count = 0;
  if (row < H_in && col < W_in) {
    for (int c=0; c<C_in; c++){
      for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
          float val = 0.0f;
          if (col + j >= 0 && col + j < W_in && row + i >= 0 && row + i < H_in) {
            val  = image[c*(H_in*W_in) + (row+i) * W_in + (col+j)];
          }
          im2col[IDX2R(idx_count, row * W_in + col, im2col_W)] = val;
          idx_count++;
        }
      }
    }
  }
}
#+END_SRC

*** Example: GEMM with Tensor Cores (WMMA API)
#+BEGIN_SRC cuda
__global__ void gemm_tensor(half *a, half *b, float *c, int M, int N, int K,
                            float alpha, float beta) {
  // ... (see gemm_tensors.cu for full implementation)
}
#+END_SRC

*** Project Plan & Status

+ *Implemented*:
  + Naive convolution (2D/3D) in RAW CUDA
  + Shared memory tiled convolution
  + im2col transformation (CPU & GPU)
  + GEMM using Tensor cores (WMMA API)
+ *TODO*:
  1. Baseline comparison with cuDNN's convolution
  2. Implement WMMA convolution as GEMM with im2col
  3. Make it an executable/library for Linux
  4. PyTorch importable module

*** Usage

+ Build with ~nvcc~ and CUDA >= 10.0
+ Input image and kernel shapes read from text files
+ Output: im2col matrix, convolution result, profiling data

*** References

+ CUDA WMMA API documentation
+ NVIDIA Nsight profiling tools
+ cuDNN library
+ NumPy ~convolve~ for correctness

*** Contact

+ For questions or contributions, open an issue or pull request.

*** License

+ MIT License

