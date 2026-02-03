* A convolutional layer from scratch in CUDA, optimize for Ada Lovelace architecture focusing on tensor cores 

Stage 1: Naive Implementation of a 3D convolutional kernel in CUDA
It uses Global Memory
Profile: High Latency and Low Occupancy

Stage 2: Basic Optimizations 
Constant memory for the Kernel (since the data doesn't change during execution)
Shared Memory Tiling. Reduces global accessess dramatically 11x fewer
Profile these
