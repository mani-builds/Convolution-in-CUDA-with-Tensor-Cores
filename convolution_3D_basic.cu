# include <stdio.h>

__global__
void convolution_3D_basic_kernel(float *N, float *F, float *P, int r, int width, int height, int depth){
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  int outDepth = blockDim.z * blockIdx.z + threadIdx.z;
  float output = 0.0;
  if (outCol < width && outRow < height && outDepth < depth){
  for (int fDep=0; fDep <2*r+1; fDep++){
    for (int fRow=0; fRow <2*r+1; fRow++){
      for (int fCol=0; fCol <2*r+1; fCol++){
        int inRow = outRow -r + fRow;
        int inCol = outCol -r + fCol;
        int inDep = outDepth -r + fDep;
        if (inRow >= 0 && inRow < height && inCol >= 0 &&
            inCol < width
            && inDep >=0 && inDep < depth){
          output += N[inDep * height * width + inRow * width + inCol]
                    * F[fDep*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol];
            }
        }
      }
    }

    P[outDepth*height*width + outRow*width + outCol] = output;

  }
}

int main(){
  float *N_h, *F_h, *P_h;
  int r, width, height, depth ;
  width = 8;
  height = 8;
  depth = 8;
  r = 1;
  N_h = (float *)malloc(sizeof(float) * width * height * depth );
  P_h = (float *)malloc(sizeof(float) * width * height * depth );
  F_h = (float *)malloc(sizeof(float) * (2*r+1) * (2*r+1) * (2*r+1));

  int count = 0;

  for (int i = 0; i < width*height*depth; i++){
    N_h[i] = count++;
  }

  for (int j=0; j< (2*r+1)*(2*r+1)*(2*r+1) ; j++){
    if (j % 2 == 0){ F_h[j] = 0;}
    else { F_h[j] = 1;}
  }

  printf("\nInput: \n");
  for (int i = 0; i < width*height*depth; i++){
    printf("%f \t", N_h[i]);
  }
  // for (int k = 0; k < depth; k++){
  // for (int j = 0; j < height; j++){
  // for (int i = 0; i < width; i++){
  //     printf("%f \t", N_h[k*depth*width + j*width + i]);
  // }}}
  printf("\nFilter: \n");
  for (int i = 0; i < (2*r+1)*(2*r+1)*(2*r+1); i++){
    printf("%f \t", F_h[i]);
  }

  // printf("\nFilter 2: \n");
// for (int fDep=0; fDep <2*r+1; fDep++){
//     for (int fRow=0; fRow <2*r+1; fRow++){
//       for (int fCol=0; fCol <2*r+1; fCol++){
//         printf("%f \t", F_h[fDep*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol]);
//       }}}


  float *N, *F, *P;
  cudaMalloc(&N, sizeof(float) * width * height * depth);
  cudaMalloc(&P, sizeof(float) * width * height * depth);
  cudaMalloc(&F, sizeof(float) * (2*r+1) * (2*r+1)* (2*r+1));

  cudaMemcpy(N, N_h,sizeof(float) * width * height * depth, cudaMemcpyHostToDevice);
  cudaMemcpy(F, F_h,sizeof(float) * (2*r+1) * (2*r+1) * (2*r+1), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(4,4,4);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                     (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);
  convolution_3D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,F,P,r,width,height,depth);

  cudaMemcpy(P_h, P,sizeof(float) * width * height * depth, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height * depth; i++) {
        printf("%f\t", P_h[i]);
    }

        printf("\n");

    cudaFree(P);
    cudaFree(N);
    cudaFree(F);
    free(F_h);
    free(N_h);
    free(P_h);


  return 0;
}
