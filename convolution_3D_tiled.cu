# include <stdio.h>

#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

#define IN_TILE_DIM 8
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height, int depth){
  int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
  int dep = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
  // loading input tiles
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  if (row >=0 && row < height && col >=0 && col <width && dep >=0 && dep < depth){
    N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*height*width + row*width + col];
}
  else{
    N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
__syncthreads();
// calculating output threads
int tileCol = threadIdx.x - FILTER_RADIUS;
int tileRow = threadIdx.y - FILTER_RADIUS;
int tileDep  = threadIdx.z - FILTER_RADIUS;
// turning off the threads at the edge
  if (row >=0 && row < height && col >=0 && col <width && dep >=0 && dep < depth){
    if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 &&
        tileCol < OUT_TILE_DIM
        && tileDep >=0 && tileDep < OUT_TILE_DIM){
    float Pvalue = 0.0f;

    for (int fDep = 0; fDep<2*FILTER_RADIUS+1; fDep++){
    for (int fRow = 0; fRow<2*FILTER_RADIUS+1; fRow++){
      for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++){
        Pvalue += F[fDep][fRow][fCol] * N_s[tileDep+fDep][tileRow+fRow][tileCol+fCol];
      }
    }
    }
    P[dep*height*width + row*width + col] = Pvalue;
  }
  }


}

int main(){
  float *N_h, *P_h;
  int width, height, depth;
  int r = FILTER_RADIUS;
  float F_h[2*r+1][2*r+1][2*r+1];

  width = 16;
  height = 16;
  depth = 16;

  N_h = (float *)malloc(sizeof(float) * width * height * depth);
  P_h = (float *)malloc(sizeof(float) * width * height* depth);

  // Create the 8x8 square of 1s in the middle (rows 4-11, cols 4-11)
  for (int d = 4; d < 12; d++) {
    for (int i = 4; i < 12; i++) {
        for (int c = 4; c < 12; c++) {
            N_h[d*height*width + i * width + c] = 1.0f;
        }
    }
  }

  // 5x5 Filter Matrix (Box Blur)
  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
  for (int k=0; k< (2*FILTER_RADIUS +1); k++){
   F_h[i][j][k] = 1.0f / 25.0f;;
  }
  }}

  // printf("\nInput: \n");
  // for (int i = 0; i < width*height; i++){
  //   printf("%f \t", N_h[i]);
  // }
  printf("\nFilter: \n");
  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
  for (int k=0; k< (2*FILTER_RADIUS +1); k++){
    printf("%f \t", F_h[i][j][k]);
  }
  }
  }

  float *N, *P;
  cudaMalloc(&N, sizeof(float) * width * height * depth);
  cudaMalloc(&P, sizeof(float) * width * height * depth);

  cudaMemcpy(N, N_h,sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F,F_h,sizeof(float) * (2*r+1) * (2*r+1) * (2*r+1));

  dim3 threadsPerBlock(IN_TILE_DIM,IN_TILE_DIM, IN_TILE_DIM);
  dim3 blocksPerGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                       (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                       (depth + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
  convolution_tiled_2D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,P,width,height, depth);
  cudaMemcpy(P_h, P,sizeof(float) * width * height * depth, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height * depth; i++) {
    if (i % width == 0){printf("\n"); }
    printf("%f\t", P_h[i]);
    }
    printf("\n");
    cudaFree(P);
    cudaFree(N);
    free(N_h);
    free(P_h);


  return 0;
}
