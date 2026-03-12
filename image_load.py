from PIL import Image
import numpy as np

def image_load():
    im_file = input("Enter the name of the image (in jpg): ")
    jpg = Image.open(im_file)
    array = np.asarray(jpg)
    print(f"Fromat {jpg.format}, size: {jpg.size}, mode: {jpg.mode}")
    print("Shape: ", array.shape)
    height, width, n_channels = array.shape[0],array.shape[1],array.shape[2]
    print("Channels, height, width: ",n_channels, height, width)
    print("Image array", array)

    array_reshaped = array.reshape(-1)

    print("Array reshaped for saving : ", array_reshaped)

    array_file = f'{im_file[:-4]}_array.txt'
    shape_file = f'{im_file[:-4]}_array_shape.txt'
    with open(array_file, 'w') as file:
        np.savetxt(file, array_reshaped, delimiter=',', fmt='%.1f')

    with open(shape_file, 'w') as fname:
        np.savetxt(fname, np.array([n_channels, height, width]), delimiter=',', fmt='%.1f')

def kernel_load():
    #kernel
    identity_4d = np.zeros((3, 3, 3, 3))
    for i in range(3):
        identity_4d[i, i, 1, 1] = 1.0 # Channel i looks only at Channel i input

    with open('kernel_array_shape.txt', 'w') as file:
        np.savetxt(file, np.array((identity_4d.shape[0], identity_4d.shape[1], identity_4d.shape[2], identity_4d.shape[3])), delimiter=',', fmt='%.1f')

    identity_4d_reshaped = identity_4d.reshape(-1)
    with open('identity_kernel_array.txt', 'w') as file:
        np.savetxt(file, identity_4d_reshaped, delimiter=',', fmt='%.1f')

    # Write the kernels into txt files
    gauss_3d = np.array([[[1,2,1],[2,4,2],[1,2,1]],
                        [[2,4,2],[4,8,4],[2,4,2]],
                        [[1,2,1],[2,4,2],[1,2,1]]])
    gauss_3d = gauss_3d / gauss_3d.sum() # Normalize so brightness stays same
    # 1. Gaussian (Wrap the 3D cube into a 4D array with 1 output channel)
    gauss_4d = np.stack([gauss_3d, gauss_3d, gauss_3d], axis=0)

    # 2. Laplacian (Already 4D)
    laplace_4d = np.full((3, 3, 3, 3), -1.0)
    for i in range(3):
        laplace_4d[i, i, 1, 1] = 26.0 # Strong center for each output channel


    with open("laplace_kernel.txt", "w") as f:
        np.savetxt(f, laplace_4d.reshape(-1), delimiter=',', fmt='%2.1f')

    with open("gaussian_kernel.txt", "w") as f:
        np.savetxt(f, gauss_4d.reshape(-1), delimiter=',', fmt='%2.6f')

if __name__ == '__main__':
    image_load()
    kernel_load()
