#!/usr/bin/env python

from PIL import Image
import numpy as np

def image_build():
    # f_name = input("Enter the file for output of gemm in text: ")
    # a_name = input("Enter the file for shape of the input array in text: ")
    # o_name = input("Enter the file for output in jpg: ")
    f_name = 'output_gemm_identity.txt'
    a_name = 'dog_array_shape.txt'
    o_name = 'image_output_identity.jpg'
    with open(f_name, 'r') as f:
        output = np.loadtxt(f)
    with open(a_name, 'r') as a:
        shape = np.loadtxt(a)
    n_channels, height, width = shape.astype(int)
    output = output.reshape([height, width, n_channels])
    output = output.astype(np.uint8)
    im_o = Image.fromarray(output)
    im_o.save(o_name)

image_build()
