// ref: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/

#include <cstdlib>
#include <stdio.h>
#include <mma.h>
#include <algorithm>

using namespace nvcuda;

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_K = 16;
const int WMMA_N = 16;


__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K,
                             float alpha, float beta) {
  // Leading dimensions
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Note: WMMA (Warp Matrix Multiply-Accumulate) is a warp-level operation, not a thread-level one.
  // The strategy is to have a single warp responsible for a single 16 x 16 section
  // of the output matrix

  // Tile using a 2D grid
  // warpSize is 32, so converting Global Thread ID into Global Warp ID  for M-dimension
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  // we don't divide the N-dimension by warpSize due to how blockDim.x and blockDim.y are set (128,4)
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // MMA's are Warp-level operations
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Set acc_frag to 0.0
  wmma::fill_fragment(acc_frag, 0.0f);

  // The strategy for the GEMM is to compute one tile of 16 x 16 output matrix per warp.
  // Loop over the K-dimension (Check out the image: 'Why loop over K.jpeg')
  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if ( aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      // The load_matrix_sync function is hardcoded to go the memory address of pointer
      // and grad a whole 16 x 16 block, and distribute among the registers of
      // 32 threads in the warp
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda , lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication and returns the 16 x 16 output tile
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      // _sync suggests to wait until all 32 threads in a warp have finished

    }
  }

    // Scale with alpha and beta
    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

      for (int i = 0; i < c_frag.num_elements; i++) {
        // Element wise addition
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }

}

int main() {

  half *a_fp16_host;
  half *b_fp16_host;
  float *c_host;

  half *a_fp16;
  half *b_fp16;
  float *c;

  float alpha = 1.0f;
  float beta = 1.0f;

  cudaEvent_t startWMMA;
  cudaEvent_t stopWMMA;

  cudaErrCheck(cudaEventCreate(&startWMMA));
  cudaErrCheck(cudaEventCreate(&stopWMMA));

  a_fp16_host = (half *)malloc(MATRIX_M * MATRIX_K * sizeof(half));
  b_fp16_host = (half *)malloc(MATRIX_K * MATRIX_N * sizeof(half));
  c_host = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  if (!a_fp16_host || !b_fp16_host || !c_host) {
    printf("host memory allocation failed\n");
    return EXIT_FAILURE;
  }

  std::fill(a_fp16_host, a_fp16_host + MATRIX_M * MATRIX_K, 1.0f);
  std::fill(b_fp16_host, b_fp16_host + MATRIX_K * MATRIX_N, 2.0f);
  std::fill(c_host, c_host + MATRIX_M * MATRIX_N, 10.0f);

  printf("a[0]: %f \n", __half2float(a_fp16_host[0]));
  printf("b[0]: %f \n", __half2float(b_fp16_host[0]));
  printf("c_host[0]: %f \n", __half2float(c_host[0]));

  cudaErrCheck(cudaMalloc(&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc(&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc(&c, MATRIX_M * MATRIX_N * sizeof(float)));

  cudaErrCheck(cudaMemcpy(a_fp16, a_fp16_host, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(b_fp16, b_fp16_host, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(c, c_host, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

  printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

  // Using WMMA
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multiple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  // Total no. of warps = 16384 * 16384 / 16 * 16 =  1024 * 1024
  // each gridDim = 256
  gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x/32 - 1))/ (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  printf("Running with wmma...\n");
  cudaErrCheck(cudaEventRecord(startWMMA));
  wmma_example<<<gridDim, blockDim>>>(a_fp16, b_fp16, c ,MATRIX_M, MATRIX_N,MATRIX_K, alpha, beta);
  cudaErrCheck(cudaEventRecord(stopWMMA));

  float *c_output;
  c_output = (float *)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  cudaErrCheck(cudaMemcpy(c_output, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost))

  printf("c_ouptut[0]: %f \n", c_output[0]);

  float wmmaTime;
  cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
  printf("wmma tokk %fms\n", wmmaTime);

  cudaErrCheck(cudaEventDestroy(startWMMA));
  cudaErrCheck(cudaEventDestroy(stopWMMA));

  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));
  cudaErrCheck(cudaFree(c));
  free(a_fp16_host);
  free(b_fp16_host);
  free(c_host);
  return 0;
}
