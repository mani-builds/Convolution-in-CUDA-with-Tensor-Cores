# GPU-Optimized Convolution from Scratch

Project Description: This project implements 2D and 3D convolutional layers from scratch in CUDA, optimized for modern GPU architectures with tensor cores. It includes naive and optimized implementations, im2col transformation, and GEMM using Tensor cores (WMMA API), with profiling and comparison against cuDNN.

## 🚀 Features

### Naive Implementation

2D/3D convolution kernel in CUDA using global memory.

Profile: High latency, low occupancy.

### Basic Optimizations

Constant Memory: Used for kernels since kernel data is static during execution.

Shared Memory Tiling: Reduces global memory accesses by approximately ~11x.

Profiling: Bottleneck analysis using NVIDIA Nsight.

### im2col Transformation

Converts input images to column format for efficient GEMM.

Implemented in Raw CUDA for GPU and NumPy for CPU verification.

### Tensor Core GEMM

Uses CUDA WMMA API for FP16 tensor operations.

Leverages warp-level matrix multiply-accumulate for high throughput.

Comparison with cuDNN: Baseline performance and correctness checks.

Executable & Library: Can be built as a Linux executable and linkable library.

PyTorch Integration: (Planned) Importable C++/CUDA module.

## 🛠 Implementation Details

### im2col

Converts input image to a padded matrix for GEMM. The GPU kernel is implemented in im2col.cu. For a $5 \times 5$ input and $3 \times 3$ kernel, im2col produces a $9 \times 25$ matrix.

### GEMM with Tensor Cores

Implemented in gemm_tensors.cu using the CUDA WMMA API for warp-level matrix multiplication. It supports large matrices (multiples of 16) and utilizes hardware-level acceleration.

### Profiling & Testing

Profiling: Nsight used to monitor occupancy, latency, and memory access patterns.

Testing: Validated with synthetic data and MNIST.

Verification: Results are cross-checked against NumPy's convolve for mathematical correctness.

## 💻 Code Examples

### im2col Kernel (CUDA)

```cuda
__global__ void im2col_kernel(float *image, int C_in, int H_in, int W_in, int im2col_H, int im2col_W, int kernel_size, float *im2col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_count = 0;

    if (row < H_in && col < W_in) {
        for (int c = 0; c < C_in; c++) {
            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                    float val = 0.0f;
                    if (col + j >= 0 && col + j < W_in && row + i >= 0 && row + i < H_in) {
                        val = image[c * (H_in * W_in) + (row + i) * W_in + (col + j)];
                    }
                    im2col[IDX2R(idx_count, row * W_in + col, im2col_W)] = val;
                    idx_count++;
                }
            }
        }
    }
}
```

### GEMM with Tensor Cores (WMMA API)

```cuda
__global__ void gemm_tensor(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
    // ... (see gemm_tensors.cu for full implementation)
    // Uses nvcuda::wmma for warp-level matrix operations
}
```

## 📅 Project Plan & Status

### [x] Implemented:

Naive convolution (2D/3D) in RAW CUDA.

Shared memory tiled convolution.

im2col transformation (CPU & GPU).

GEMM using Tensor cores (WMMA API).

### [ ] TODO:

Baseline comparison with cuDNN's convolution.

Implement WMMA convolution as GEMM with im2col.

Finalize Linux build system (CMake/Make).

Create PyTorch importable module.

## 📦 Usage

Build: Use nvcc and ensure CUDA >= 10.0 is installed.

Input: Shapes are read from text files or provided via command line arguments.

Output: Generates the im2col matrix, convolution result, and detailed profiling data.

## 📚 References

CUDA WMMA API Documentation

NVIDIA Nsight Profiling Tools

cuDNN Library

## 🤝 Contact

For questions or contributions, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.