#include <stdio.h>

__global__ void convolution_gemm_kernel(float *kernel, float *im2col, float *output, int C_out,
                                        int patch_size, int number_of_patches) {

  // We perform matmul of sizes (Cout​×PatchSize)×(PatchSize×NumPatches)
  // The output shape is (Cout x NumPatches), so each thread process one element
  // of the output matrix.


  // For eg: if output shape is 3 x 50240
  int out_chan = blockDim.y * blockIdx.y +  threadIdx.y; // Goes from 0 to 2 (C_out)
  int patch_id = blockDim.x * blockIdx.x + threadIdx.x; // Goes from 0 to 50239 (NumPatches)

  if (out_chan < C_out && patch_id < number_of_patches) {
    float pixel_sum = 0.0f;
    for (int i = 0; i < patch_size; i++) {
      pixel_sum += kernel[out_chan * patch_size + i] * im2col[i* number_of_patches + patch_id];
    }
    output[out_chan * number_of_patches + patch_id] = pixel_sum;
  }

}


int main() {

  float *im2col_h;
  float *kernel_h;
  float *output_h;

  // Reading Kernel size

  FILE *kfile = fopen("kernel_array_shape.txt", "r");
  if (kfile == NULL) {
        return 1; // Error opening file
    }
    float svalue;
    float *kernel_shape;
    kernel_shape = (float *)malloc(sizeof(float)*4);
    int i =0;
    while (fscanf(kfile, "%f", &svalue) == 1) {
      kernel_shape[i] = svalue;
        i++;
    }


  int C_out = kernel_shape[0];
  int patch_size = kernel_shape[1]*kernel_shape[2]*kernel_shape[3];

  // Reading output image shape
  FILE *file = fopen("dog_array_shape.txt", "r");
    if (file == NULL) {
        return 1; // Error opening file
    }
    float value;
    float *image_shape;
    image_shape = (float *)malloc(sizeof(float)*3);
    i =0;
    while (fscanf(file, "%f", &value) == 1) {
      image_shape[i] = value;
        i++;
    }

    int C_in = image_shape[0];
    int number_of_patches = image_shape[1] * image_shape[2];
    // Reading Kernel matrix (in 1D array)

  FILE *kernel_file = fopen("identity_kernel_array.txt", "r");
  if (kernel_file == NULL) {
        return 1; // Error opening file
    }
    i=0;
    float kvalue;
    kernel_h = (float *)malloc(C_out*patch_size*sizeof(float));
    while (fscanf(kernel_file, "%f", &kvalue) == 1) {
      kernel_h[i] = kvalue;
      i++;
    }

    printf("C_out: %d, patch_size: %d, number_of_patches: %d\n", C_out,
           patch_size,
           number_of_patches);

    // Reading the im2col matrix (in 1D array form)
    im2col_h = (float *)malloc(patch_size*number_of_patches*sizeof(float));
    FILE *im2col_file = fopen("im2col_array_identity.txt", "r");
    if (im2col_file == NULL) {
      return 1;
    }
    i = 0;
    float im2col_value;
    while (fscanf(im2col_file, "%f", &im2col_value) == 1) {
      im2col_h[i] = im2col_value;
      i++;
    }

    float *im2col;
    float *kernel;
    float *output;

    cudaMalloc(&im2col, patch_size*number_of_patches*sizeof(float));
    cudaMalloc(&kernel, C_out*patch_size*sizeof(float));
    cudaMalloc(&output, C_out * number_of_patches * sizeof(float));

    cudaMemcpy(im2col, im2col_h, patch_size * number_of_patches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel, kernel_h, C_out * patch_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid(
        (number_of_patches + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (C_out + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolution_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>
    (kernel, im2col, output, C_out,patch_size, number_of_patches);

    output_h = (float *)malloc(C_out * number_of_patches * sizeof(float));

    cudaMemcpy(output_h, output, C_out * number_of_patches * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *ofile = fopen("output_gemm_identity.txt", "w");
    if (ofile == NULL){ return 1;}
    for (int i = 0; i<C_out * number_of_patches; i++){
      fprintf(ofile, "%.0f ", output_h[i]);
    }

    fclose(kernel_file);
    fclose(file);
    fclose(kfile);
    fclose(im2col_file);
    fclose(ofile);
    free(kernel_shape);
    free(image_shape);
    free(kernel_h);
    free(im2col_h);
    free(output_h);
    cudaFree(im2col);
    cudaFree(kernel);
    cudaFree(output);

  return 0;

}
