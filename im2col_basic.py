import numpy as np
import math

def dim_output(size_of_input, P, K, S):
    return math.floor((size_of_input + 2*P - K) / S ) + 1

image = np.array([i for i in range(0,3*5*5)])
image = image.reshape([3,5,5]) # channel, row, col

# This is a single filter and the output will be Grayscale version of input
filter_unshaped = np.array([i for i in range(0,3*3*3)])
filter = filter_unshaped.reshape([3,3,3])
print(f"image's shape: {image.shape}")
print(image)
print(f"filter's shape: {filter.shape}")
print(filter)

# 3D filter, input and output will be in RGB
# Create 3 different filters (3 filters, each 3x3x3)
filters_3d = np.random.randn(3,3,3,3)


stride = 1 # how many pixels the filter jumps each time
kernel_radius = filter.shape[1]//2
padding = kernel_radius #(filter.shape[1] - 1) / 2 # the number of zero-pixel borders added

C_in = image.shape[0]
H_in = image.shape[1]
W_in = image.shape[2]

print(f"IN DIMS: {H_in}, {W_in}")
kernel_size = filter.shape[1]
patch_size = filter.shape[1] * filter.shape[2] * filter.shape[0]

C_out = C_in
H_out = dim_output(H_in, padding, kernel_size, stride) # No. of position for Height
W_out = dim_output(W_in, padding, kernel_size, stride) # No. of position for Width

print(f"OUT DIMS: {H_out}, {W_out}")
num_of_patches = H_out * W_out

# 1. Pad the image: (Channels, Height, Width)
# We only want to pad the spatial dimensions (H and W), not the channels.
# ((0,0), (1,1), (1,1)) means 0 padding for channels, 1 for H, 1 for W.
image_padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

im2col_image = np.zeros([patch_size, num_of_patches])
# (patch_size*num_of_patches)
# im2col_image = im2col_image.reshape

# Flatten input image
col_idx = 0
for i in range(H_out):
    for j in range(W_out):
        start_h = i * stride
        start_w = j * stride

        # Now fill the column for this specific patch
        idx_count = 0
        for channel in range(filter.shape[0]):
            for row in range(filter.shape[1]):
                for col in range(filter.shape[2]):
                    im2col_image[idx_count][col_idx] = image_padded[channel, start_h + row, start_w + col]
                    idx_count += 1
        col_idx += 1
# for each_patch in range(num_of_patches):
#     # 1. Calculate the top-left corner of the current patch
#     # This converts 1D patch index into 2D (row, col) coordinates
#     start_h = (each_patch // W_out) * stride
#     start_w = (each_patch % W_out) * stride

#     idx_count = 0
#     for channel in range(filter.shape[0]):
#         for row in range(filter.shape[1]):
#             for col in range(filter.shape[2]):
#                 # 2. Use the start coordinates to "offset" the selection
#                 im2col_image[idx_count][each_patch] = image_padded[channel, start_h + row, start_w + col]
#                 idx_count += 1

print(im2col_image)
print("Shape of im2col_image: ", im2col_image.shape)

# Flatten kernel
filter_flattened = filter.reshape(1, -1)
# filter_broadcasted = np.array([filter_unshaped for i in range(len(filter_unshaped))])
#print("Shape of filter_boardcasted: ", filter_broadcasted.shape)
print("Shape of filter_flattened: ", filter_flattened.shape)
print(filter_flattened)

# output = filter_broadcasted * im2col_image
output = np.matmul(filter_flattened, im2col_image)
output = output.reshape([H_out,W_out])
print("Output: \n", output)
print("Shape of output: ", output.shape)


# 3d filter and its output
# Flatten to (Number of filters, total pixels per filter)
filter_matrix = filters_3d.reshape(3, -1) # Shape: (3, 27)
print("Shape of filter_matrix: ", filter_matrix.shape)
# print(filter_flattened)

# Matrix Multiplication
output_3d = np.matmul(filter_matrix, im2col_image) # Shape: (3, 9)

# Reshape to (Channels, Height, Width)
output_3d_spatial = output_3d.reshape(3, H_out, W_out) # Shape: (3, 3, 3)
print("Shape of output3d_spatial: ", output_3d_spatial.shape)
# print("kernel_radius: ", kernel_radius)
