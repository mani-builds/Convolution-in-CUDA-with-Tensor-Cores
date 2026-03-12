import numpy as np
import math

def dim_output(size_of_input, P, K, S):
    return math.floor((size_of_input + 2*P - K) / S ) + 1


image = np.loadtxt('dog_array.txt')
image_shape = np.loadtxt('dog_array_shape.txt')

C_in = int(image_shape[0])
H_in = int(image_shape[1])
W_in = int(image_shape[2])

image = image.reshape((C_in,H_in,W_in))

# 1. Gaussian-like Blur Kernel (Approximation)
# We create one 3x3x3 cube and repeat it for all 3 input channels
gauss_3d = np.array([
    [[1,2,1],[2,4,2],[1,2,1]], # Slice 0
    [[2,4,2],[4,8,4],[2,4,2]], # Slice 1 (Center)
    [[1,2,1],[2,4,2],[1,2,1]]  # Slice 2
])
gauss_3d = gauss_3d / gauss_3d.sum() # Normalize so brightness stays same
# 1. Gaussian (Wrap the 3D cube into a 4D array with 1 output channel)
gauss_4d = np.stack([gauss_3d, gauss_3d, gauss_3d], axis=0)

# 2. Laplacian (Already 4D)
laplace_4d = np.full((3, 3, 3, 3), -1.0)
for i in range(3):
    laplace_4d[i, i, 1, 1] = 26.0 # Strong center for each output channel

# 3. Identity (Fixed to pass all 3 channels through)
identity_4d = np.zeros((3, 3, 3, 3))
for i in range(3):
    identity_4d[i, i, 1, 1] = 1.0 # Channel i looks only at Channel i input


# CHOOSE YOUR FILTER HERE
filter = identity_4d
C_out = filter.shape[0] # Dynamically set output channels (1 for gauss, 3 for others)
print(f"image's shape: {image.shape}")
print(image)

stride = 1 # how many pixels the filter jumps each time
kernel_radius = filter.shape[1]//2
padding = kernel_radius #(filter.shape[1] - 1) / 2 # the number of zero-pixel borders added

print(f"IN DIMS: {C_in}, {H_in}, {W_in}")
kernel_size = filter.shape[1]
patch_size = filter.shape[1] * filter.shape[2] * filter.shape[3]

H_out = dim_output(H_in, padding, kernel_size, stride) # No. of position for Height
W_out = dim_output(W_in, padding, kernel_size, stride) # No. of position for Width

print(f"OUT DIMS: {C_out}, {H_out}, {W_out}")
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
# for c in range(C_in):
for i in range(H_out):
    for j in range(W_out):
        start_h = i * stride
        start_w = j * stride

        # Now fill the column for this specific patch
        idx_count = 0
        for channel in range(filter.shape[1]):
            for row in range(filter.shape[2]):
                for col in range(filter.shape[3]):
                    im2col_image[idx_count][col_idx] = image_padded[channel, start_h + row, start_w + col]
                    idx_count += 1
        col_idx += 1

print(im2col_image)
print("Shape of im2col_image: ", im2col_image.shape)

with open("im2col_array_py.txt", "w") as file:
    np.savetxt(file, im2col_image, fmt="%.0f" )
print("im2col_image is saved at {}")
# Flatten kernel
filter_flattened = filter.reshape(C_out, -1)
# filter_broadcasted = np.array([filter_unshaped for i in range(len(filter_unshaped))])
#print("Shape of filter_boardcasted: ", filter_broadcasted.shape)
print("Shape of filter_flattened: ", filter_flattened.shape)
print(filter_flattened)

# output = filter_broadcasted * im2col_image
output = np.matmul(filter_flattened, im2col_image)
output = output.reshape([C_out, H_out,W_out])
print("Output: \n", output)
print("Shape of output: ", output.shape)
